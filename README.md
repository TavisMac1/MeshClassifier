## Classification Examples

<img width="2857" height="1535" alt="car_body" src="https://github.com/user-attachments/assets/cd12096c-10ee-4480-88ec-49de2b34a569" />
<img width="2857" height="1276" alt="anchor" src="https://github.com/user-attachments/assets/5a6f286f-ae4f-434e-86da-c9f15d1aee5e" />
<img width="2823" height="1248" alt="trex" src="https://github.com/user-attachments/assets/74c2dc04-1aa3-48da-a112-838f42ea32bd" />
<img width="2826" height="1220" alt="sword" src="https://github.com/user-attachments/assets/8e4e1d91-ce3a-425a-ae44-3157b36be6c7" />
<img width="2865" height="1225" alt="cat" src="https://github.com/user-attachments/assets/583f16bf-6087-4d3f-825e-658c123664a9" />
# AI Mesh Classifier

Classify and name a selected Blender mesh with a local vision-language model. The add-on renders the mesh, sends that image to a local service, and writes the returned name, category, and description onto the object. Images are not sent to a hosted API.

## Requirements

- Windows x64, Blender 4.2 or newer, and an AMD GPU with a working Vulkan driver
- Python 3.10 installed with the Windows Python Launcher (`py -3.10`)
- Internet access for the initial dependency, runtime, and model downloads

The add-on keeps its service environment separate from Blender's bundled Python. It installs the service dependencies into Blender's user configuration folder and starts the service itself. Python 3.10 does not have to be selected as Blender's interpreter.

The first setup installs lightweight service dependencies. The first classification downloads the llama.cpp Vulkan runtime (38 MB) and the quantized SmolVLM-500M model and vision projector (546 MB total). Downloads are verified by SHA-256 and reused on later launches. The old Qwen model used about 7.5 GB for its weights alone. SmolVLM is substantially smaller, which can reduce recognition accuracy, particularly for specialized assets. Once downloaded, classification can run offline.

## Quick Blender test

1. Build the installable add-on ZIP from this project directory:

   ```powershell
   py -3.10 tools\build_addon.py
   ```

2. In Blender, open **Edit > Preferences > Add-ons > Install from Disk**, select `dist\mesh_classifier.zip`, and enable **Mesh Classifier**.
3. In a 3D View, press **Shift+A > Mesh > UV Sphere** (or use the default cube).
4. Open the 3D View sidebar with **N**, then select **Mesh Classifier**.
5. Click **Install / Start AI Service**. Leave Blender open while it installs dependencies; the panel shows its status. The first setup may take several minutes. If setup fails, read `setup.log` in Blender's user configuration folder.
6. In the panel, set **Render Views** (default 3). Higher values can improve recognition for ambiguous meshes at the cost of extra render/inference time.
7. When the status says **Ready**, select the mesh and click **Classify Selected Mesh**. The panel reports actual download bytes and progress, model loading, the detected Vulkan GPU, classification, and elapsed time. Model loading happens once per service launch; subsequent requests reuse the loaded model.
8. Check the object's new name and the **Category** and **Description** shown in the panel. The category and description are also stored as the object's `ai_category` and `ai_description` custom properties.

The service listens only on `127.0.0.1`. It runs while Blender is open and is stopped when the add-on is disabled or Blender exits. The add-on does not modify Blender's Python installation.

## Troubleshooting

- If classification reports `WinError 10061` / “actively refused,” update or reinstall the latest add-on ZIP and click **Install / Start AI Service** again. The service startup log is `service.log`; a successful health check means the service is listening.
- After upgrading the add-on, disable the old add-on before installing the new ZIP so its service is stopped. If the panel reports an older service is still running, close Blender and stop the old service before starting the updated add-on. Stopping a running service interrupts its current classification.
- If setup says the `py` launcher is missing, install Python 3.10 from python.org with the Python Launcher option enabled, then restart Blender.
- Setup output is saved to `setup.log`; service startup output is saved to `service.log` in the add-on's user configuration folder.
- The new backend requires an AMD Vulkan GPU and reports an error instead of silently switching to CPU. Update the AMD graphics driver if the GPU is not detected. The service writes inference details to `vulkan-b12274\vulkan-service.log` in the add-on's user configuration folder.
- The compact model can misidentify unusual meshes and produce incorrect or cut-off descriptions. Review its output, particularly for specialized assets. In an initial local check it identified a cube and a sphere but classified Blender's Suzanne monkey as a robot.
- Existing Qwen downloads and the previous Python dependencies are not removed automatically. They are no longer used by the Vulkan backend.
- Keep the object selected and make sure it is a mesh. The add-on renders it in isolation before classification.

## Development

Run the tests without downloading the model (requires the service dependencies):

```powershell
python -m unittest discover -s tests -v
```

Rebuild the add-on ZIP after changing either `blender_addon` or `service`:

```powershell
py -3.10 tools\build_addon.py
```
