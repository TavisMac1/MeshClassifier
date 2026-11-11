"""Blender integration for Mesh Classifier."""

bl_info = {
    "name": "Mesh Classifier",
    "author": "Mesh Classifier contributors",
    "version": (0, 1, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Sidebar > Mesh Classifier",
    "description": "Name and categorize a selected mesh using a local vision model",
    "category": "Object",
}

import json
import math
import os
from pathlib import Path
import queue
import shutil
import subprocess
import threading
import time
import tempfile
import urllib.error
import urllib.request

import bpy
from mathutils import Vector


_SETUP_RESULTS = queue.Queue()
_SETUP_THREAD = None
_SERVICE_PROCESS = None
_SERVICE_STATUS = "Not started"
_SERVICE_STARTED_AT = None
_CLASSIFICATION_STATUS = []


def _service_is_compatible(url):
    with urllib.request.urlopen(url.rstrip("/") + "/health", timeout=0.5) as response:
        health = json.loads(response.read().decode("utf-8"))
    if health.get("progress_api") != 3:
        raise ValueError(
            "An older AI service is running. Disable the old add-on or restart Blender, "
            "then start the service again."
        )
    return True


def _service_paths():
    addon_dir = Path(__file__).resolve().parent
    bundled_service = addon_dir / "runtime_service"
    if bundled_service.is_dir():
        return bundled_service, addon_dir, "runtime_service.app"

    source_service = addon_dir.parent / "service"
    if source_service.is_dir():
        return source_service, source_service.parent, "service.app"
    raise FileNotFoundError("The bundled AI service files are missing. Rebuild and reinstall the add-on ZIP.")


def _environment_paths():
    config_dir = Path(bpy.utils.user_resource("CONFIG", path="mesh_classifier", create=True))
    environment_dir = config_dir / "venv"
    if os.name == "nt":
        python_executable = environment_dir / "Scripts" / "python.exe"
    else:
        python_executable = environment_dir / "bin" / "python"
    return config_dir, environment_dir, python_executable, config_dir / "setup-vulkan-complete"


def _run_setup(config_dir, environment_dir, python_executable, setup_marker, requirements):
    launcher = shutil.which("py")
    if launcher is None:
        _SETUP_RESULTS.put(("error", "Python Launcher 'py' was not found. Install Python 3.10 and its launcher, then restart Blender."))
        return

    config_dir.mkdir(parents=True, exist_ok=True)
    log_path = config_dir / "setup.log"
    try:
        with log_path.open("w", encoding="utf-8") as log_file:
            commands = [
                [launcher, "-3.10", "-m", "venv", str(environment_dir)],
                [str(python_executable), "-m", "pip", "install", "--upgrade", "pip"],
                [str(python_executable), "-m", "pip", "install", "-r", str(requirements)],
            ]
            for command in commands:
                completed = subprocess.run(
                    command,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
                if completed.returncode:
                    _SETUP_RESULTS.put(
                        ("error", "Dependency setup failed. See the log: %s" % log_path)
                    )
                    return
        setup_marker.write_text("Dependencies installed.\n", encoding="utf-8")
    except OSError as exc:
        _SETUP_RESULTS.put(("error", "Could not set up the AI service: %s" % exc))
        return
    _SETUP_RESULTS.put(("ready", ""))


def _start_service():
    global _SERVICE_PROCESS, _SERVICE_STARTED_AT, _SERVICE_STATUS

    _service_dir, import_root, module = _service_paths()
    config_dir, _environment_dir, python_executable, _setup_marker = _environment_paths()
    log_file = (config_dir / "service.log").open("a", encoding="utf-8")
    environment = os.environ.copy()
    environment["MESH_CLASSIFIER_DATA"] = str(config_dir)
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = str(import_root) + (
        os.pathsep + existing_pythonpath if existing_pythonpath else ""
    )
    command = [
        str(python_executable),
        "-m",
        "uvicorn",
        "%s:app" % module,
        "--app-dir",
        str(import_root),
        "--host",
        "127.0.0.1",
        "--port",
        "8765",
    ]
    creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    _SERVICE_PROCESS = subprocess.Popen(
        command,
        cwd=str(import_root),
        env=environment,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        creationflags=creation_flags,
    )
    log_file.close()
    _SERVICE_STARTED_AT = time.monotonic()
    _SERVICE_STATUS = "Starting local service..."


def _poll_service_setup():
    global _SERVICE_STATUS, _SETUP_THREAD

    if not getattr(bpy.types.Scene, "mesh_classifier_status", None):
        return None

    try:
        result, detail = _SETUP_RESULTS.get_nowait()
    except queue.Empty:
        result = None
        detail = ""

    if result == "error":
        _SERVICE_STATUS = detail
        _SETUP_THREAD = None
    elif result == "ready":
        _SETUP_THREAD = None
        try:
            _start_service()
        except (FileNotFoundError, OSError, subprocess.SubprocessError) as exc:
            _SERVICE_STATUS = "Could not start service: %s" % exc
            _SETUP_THREAD = None

    if _SERVICE_STATUS == "Starting local service...":
        health_url = bpy.context.scene.mesh_classifier_url
        try:
            if _service_is_compatible(health_url):
                _SERVICE_STATUS = "Ready"
                _SETUP_THREAD = None
        except ValueError as exc:
            _SERVICE_STATUS = str(exc)
            _SETUP_THREAD = None
        except (urllib.error.URLError, TimeoutError):
            if _SERVICE_PROCESS is not None and _SERVICE_PROCESS.poll() is not None:
                _SERVICE_STATUS = "Service stopped; see service.log in the add-on data folder."
                _SETUP_THREAD = None
            elif time.monotonic() - _SERVICE_STARTED_AT > 60:
                _SERVICE_STATUS = "Service did not start; see service.log in the add-on data folder."
                _SETUP_THREAD = None

    bpy.context.scene.mesh_classifier_status = _SERVICE_STATUS
    if _SETUP_THREAD is not None or _SERVICE_STATUS == "Starting local service...":
        return 1.0
    return None


def _request_classification(url, image_paths):
    boundary = "----BlenderAIAssistantBoundary7MA4YWxkTrZu0gW"
    body_parts = []
    for index, image_path in enumerate(image_paths):
        with open(image_path, "rb") as image_file:
            image_bytes = image_file.read()
        body_parts.append(
            ("--%s\r\n" % boundary).encode("ascii")
            + ('Content-Disposition: form-data; name="files"; filename="mesh_view_%s.png"\r\n' % index).encode("ascii")
            + b"Content-Type: image/png\r\n\r\n"
            + image_bytes
            + b"\r\n"
        )
    body = b"".join(body_parts) + ("\r\n--%s--\r\n" % boundary).encode("ascii")
    request = urllib.request.Request(
        url.rstrip("/") + "/jobs",
        data=body,
        headers={"Content-Type": "multipart/form-data; boundary=%s" % boundary},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        result = json.loads(response.read().decode("utf-8"))
    job_id = result.get("id")
    if not isinstance(job_id, str) or not job_id.isalnum():
        raise ValueError("The service returned an invalid job ID.")
    return url.rstrip("/") + "/jobs/" + job_id


def _format_progress(status):
    elapsed = status.get("elapsed_seconds", 0)
    message = status.get("message", "Waiting for service.")
    lines = [message, "Elapsed: %dm %02ds" % (elapsed // 60, elapsed % 60)]
    if status.get("state") == "download":
        downloaded = status.get("downloaded_bytes")
        total_bytes = status.get("total_bytes")
        if isinstance(downloaded, int) and isinstance(total_bytes, int) and total_bytes > 0:
            lines.append("Downloaded: %.1f / %.1f MB (%.0f%%)" % (downloaded / 1e6, total_bytes / 1e6, 100 * downloaded / total_bytes))
        done = status.get("files_done")
        total = status.get("files_total")
        if isinstance(done, int) and isinstance(total, int) and total > 0:
            lines.append("Model files checked: %s / %s (%.0f%%)" % (done, total, 100 * done / total))
        cache_bytes = status.get("cache_bytes")
        if isinstance(cache_bytes, int):
            lines.append("Model cache on disk: %.2f GB (includes cached files)" % (cache_bytes / 1e9))
    return lines


def _classify_in_background(url, image_paths, results):
    try:
        job_url = _request_classification(url, image_paths)
        while True:
            with urllib.request.urlopen(job_url, timeout=15) as response:
                status = json.loads(response.read().decode("utf-8"))
            state = status.get("state")
            if state == "error":
                results.put(("error", status.get("message", "Classification failed.")))
                break
            if state == "done":
                result = status.get("result")
                if not isinstance(result, dict):
                    raise ValueError("The service returned an invalid classification.")
                for field in ("name", "category", "description"):
                    if not isinstance(result.get(field), str) or not result[field].strip():
                        raise ValueError("The service returned an invalid %s." % field)
                results.put(("success", result))
                break
            results.put(("progress", _format_progress(status)))
            time.sleep(1)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        results.put(("error", "Service returned HTTP %s: %s" % (exc.code, detail[:300])))
    except (urllib.error.URLError, TimeoutError, ValueError, OSError, RuntimeError) as exc:
        if isinstance(exc, urllib.error.URLError):
            results.put(
                (
                    "error",
                    "Local AI service is unreachable at %s. Click Install / Start AI Service; "
                    "if it still fails, inspect service.log in the add-on data folder. (%s)"
                    % (url, exc),
                )
            )
        else:
            results.put(("error", "Could not classify mesh: %s" % exc))
    except Exception as exc:
        results.put(("error", "Unexpected classification failure: %s" % exc))
    finally:
        for image_path in image_paths:
            try:
                os.remove(image_path)
            except OSError:
                pass


def _render_selected_object(scene, target, image_paths):
    original_camera = scene.camera
    original_resolution = (scene.render.resolution_x, scene.render.resolution_y, scene.render.resolution_percentage)
    original_filepath = scene.render.filepath
    original_format = scene.render.image_settings.file_format
    original_transparency = scene.render.film_transparent
    visibility = {obj: obj.hide_render for obj in scene.objects}
    temporary_objects = []

    try:
        corners = [target.matrix_world @ Vector(corner) for corner in target.bound_box]
        center = sum(corners, Vector()) / len(corners)
        radius = max((corner - center).length for corner in corners)
        radius = max(radius, 0.1)

        camera_data = bpy.data.cameras.new("Mesh Classifier Camera")
        camera = bpy.data.objects.new("Mesh Classifier Camera", camera_data)
        scene.collection.objects.link(camera)
        temporary_objects.append(camera)
        base_direction = Vector((1.7, -2.2, 1.6)).normalized()
        camera.location = center + base_direction * radius * 4.5
        camera_data.type = "ORTHO"
        camera_data.ortho_scale = radius * 2.8
        camera.rotation_euler = (center - camera.location).to_track_quat("-Z", "Y").to_euler()
        scene.camera = camera

        light_data = bpy.data.lights.new("Mesh Classifier Light", type="AREA")
        light = bpy.data.objects.new("Mesh Classifier Light", light_data)
        scene.collection.objects.link(light)
        temporary_objects.append(light)
        light.location = center + Vector((1.5, -2.0, 3.0)) * radius
        light_data.energy = max(500.0, radius * radius * 1000.0)
        light_data.shape = "DISK"
        light_data.size = radius * 2.0
        light.rotation_euler = (center - light.location).to_track_quat("-Z", "Y").to_euler()

        for obj in scene.objects:
            obj.hide_render = obj not in (target, camera, light)
        target.hide_render = False
        scene.render.resolution_x = 512
        scene.render.resolution_y = 512
        scene.render.resolution_percentage = 100
        scene.render.image_settings.file_format = "PNG"
        scene.render.film_transparent = False
        count = len(image_paths)
        for index, image_path in enumerate(image_paths):
            angle = (2 * math.pi * index) / count
            direction = Vector(
                (
                    base_direction.x * math.cos(angle) - base_direction.y * math.sin(angle),
                    base_direction.x * math.sin(angle) + base_direction.y * math.cos(angle),
                    base_direction.z,
                )
            ).normalized()
            camera.location = center + direction * radius * 4.5
            camera.rotation_euler = (center - camera.location).to_track_quat("-Z", "Y").to_euler()
            bpy.ops.render.render()
            bpy.data.images["Render Result"].save_render(image_path, scene=scene)
    finally:
        for obj in temporary_objects:
            bpy.data.objects.remove(obj, do_unlink=True)
        scene.camera = original_camera
        scene.render.resolution_x, scene.render.resolution_y, scene.render.resolution_percentage = original_resolution
        scene.render.filepath = original_filepath
        scene.render.image_settings.file_format = original_format
        scene.render.film_transparent = original_transparency
        for obj, hidden in visibility.items():
            obj.hide_render = hidden


class MESHCLASSIFIER_OT_classify(bpy.types.Operator):
    bl_idname = "mesh_classifier.classify"
    bl_label = "Classify Selected Mesh"
    bl_description = "Render the selected mesh and classify it using the local AI service"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return context.active_object is not None and context.active_object.type == "MESH"

    def execute(self, context):
        global _CLASSIFICATION_STATUS

        target = context.active_object
        url = context.scene.mesh_classifier_url
        try:
            view_count = context.scene.mesh_classifier_view_count
            image_paths = []
            for _ in range(view_count):
                image_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                image_file.close()
                image_paths.append(image_file.name)
            _render_selected_object(context.scene, target, image_paths)
        except Exception as exc:
            for image_path in locals().get("image_paths", []):
                try:
                    os.remove(image_path)
                except FileNotFoundError:
                    pass
            self.report({"ERROR"}, "Could not render mesh: %s" % exc)
            return {"CANCELLED"}

        self._results = queue.Queue()
        self._target_name = target.name
        self._timer = context.window_manager.event_timer_add(1.0, window=context.window)
        threading.Thread(
            target=_classify_in_background,
            args=(url, image_paths, self._results),
            daemon=True,
        ).start()
        context.window_manager.modal_handler_add(self)
        _CLASSIFICATION_STATUS = ["Sending %s rendered views to the local service." % len(image_paths)]
        self.report({"INFO"}, _CLASSIFICATION_STATUS[0])
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        global _CLASSIFICATION_STATUS

        if event.type == "ESC":
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None
            _CLASSIFICATION_STATUS = ["Classification continues in the background; the result will be discarded."]
            return {"CANCELLED"}
        if event.type != "TIMER":
            return {"PASS_THROUGH"}
        if context.area is not None:
            context.area.tag_redraw()

        try:
            result_type, result = self._results.get_nowait()
        except queue.Empty:
            return {"PASS_THROUGH"}

        if result_type == "progress":
            _CLASSIFICATION_STATUS = result
            return {"PASS_THROUGH"}

        context.window_manager.event_timer_remove(self._timer)
        self._timer = None
        if result_type == "error":
            _CLASSIFICATION_STATUS = [result]
            self.report({"ERROR"}, result)
            return {"CANCELLED"}

        target = bpy.data.objects.get(self._target_name)
        if target is None:
            _CLASSIFICATION_STATUS = ["Classification finished, but the original mesh was removed."]
            self.report({"WARNING"}, _CLASSIFICATION_STATUS[0])
            return {"CANCELLED"}
        target.name = result["name"]
        target["ai_category"] = result["category"]
        target["ai_description"] = result["description"]
        _CLASSIFICATION_STATUS = ["Classified as %s (%s)" % (result["name"], result["category"])]
        self.report({"INFO"}, _CLASSIFICATION_STATUS[0])
        return {"FINISHED"}

    def cancel(self, context):
        if getattr(self, "_timer", None) is not None:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None


class MESHCLASSIFIER_OT_setup_service(bpy.types.Operator):
    bl_idname = "mesh_classifier.setup_service"
    bl_label = "Install / Start AI Service"
    bl_description = "Install the local AI dependencies in a separate environment and start the service"

    def execute(self, context):
        global _SETUP_THREAD, _SERVICE_STATUS

        if _SETUP_THREAD is not None and _SETUP_THREAD.is_alive():
            self.report({"INFO"}, "AI service setup is already running.")
            return {"CANCELLED"}

        try:
            service_dir, _import_root, _module = _service_paths()
            config_dir, environment_dir, python_executable, setup_marker = _environment_paths()
            requirements = service_dir / "requirements.txt"
        except (FileNotFoundError, OSError) as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}

        if not requirements.is_file():
            self.report({"ERROR"}, "Service requirements are missing from the add-on.")
            return {"CANCELLED"}

        _SERVICE_STATUS = "Installing dependencies (first setup can take several minutes)..."
        context.scene.mesh_classifier_status = _SERVICE_STATUS
        if setup_marker.is_file() and python_executable.is_file():
            try:
                service_ready = _service_is_compatible(context.scene.mesh_classifier_url)
            except (urllib.error.URLError, TimeoutError):
                service_ready = False
            except ValueError as exc:
                _SERVICE_STATUS = str(exc)
                context.scene.mesh_classifier_status = _SERVICE_STATUS
                self.report({"ERROR"}, _SERVICE_STATUS)
                return {"CANCELLED"}
            if service_ready:
                _SERVICE_STATUS = "Ready"
            else:
                try:
                    _start_service()
                except (FileNotFoundError, OSError, subprocess.SubprocessError) as exc:
                    self.report({"ERROR"}, "Could not start the AI service: %s" % exc)
                    return {"CANCELLED"}
        else:
            _SETUP_THREAD = threading.Thread(
                target=_run_setup,
                args=(config_dir, environment_dir, python_executable, setup_marker, requirements),
                daemon=True,
            )
            _SETUP_THREAD.start()
        bpy.app.timers.register(_poll_service_setup, first_interval=1.0)
        return {"FINISHED"}


class MESHCLASSIFIER_PT_main(bpy.types.Panel):
    bl_label = "Mesh Classifier"
    bl_idname = "MESHCLASSIFIER_PT_main"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Mesh Classifier"

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        layout.prop(scene, "mesh_classifier_url", text="Service URL")
        layout.prop(scene, "mesh_classifier_view_count", text="Render Views")
        layout.operator(MESHCLASSIFIER_OT_setup_service.bl_idname, icon="PREFERENCES")
        layout.label(text="Service: %s" % scene.mesh_classifier_status)
        layout.operator(MESHCLASSIFIER_OT_classify.bl_idname, icon="OUTLINER_OB_MESH")
        if _CLASSIFICATION_STATUS:
            for line in _CLASSIFICATION_STATUS:
                layout.label(text=line)
        target = context.active_object
        if target and target.type == "MESH" and "ai_category" in target:
            layout.label(text="Category: %s" % target["ai_category"])
            layout.label(text="Description: %s" % target.get("ai_description", ""))


CLASSES = (MESHCLASSIFIER_OT_classify, MESHCLASSIFIER_OT_setup_service, MESHCLASSIFIER_PT_main)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.mesh_classifier_url = bpy.props.StringProperty(
        name="Service URL",
        description="Address of the local Mesh Classifier service",
        default="http://127.0.0.1:8765",
    )
    bpy.types.Scene.mesh_classifier_status = bpy.props.StringProperty(
        name="Mesh Classifier Service Status",
        default=_SERVICE_STATUS,
    )
    bpy.types.Scene.mesh_classifier_view_count = bpy.props.IntProperty(
        name="Render Views",
        description="How many camera angles to render and send to the classifier",
        default=3,
        min=1,
        max=6,
    )


def unregister():
    global _SERVICE_PROCESS

    if bpy.app.timers.is_registered(_poll_service_setup):
        bpy.app.timers.unregister(_poll_service_setup)
    if _SERVICE_PROCESS is not None and _SERVICE_PROCESS.poll() is None:
        _SERVICE_PROCESS.terminate()
        _SERVICE_PROCESS = None
    del bpy.types.Scene.mesh_classifier_view_count
    del bpy.types.Scene.mesh_classifier_status
    del bpy.types.Scene.mesh_classifier_url
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
