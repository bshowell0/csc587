### Overall Summary

This repository, `3d-pano-inpainting`, implements a pipeline to convert a single 360° equirectangular panoramic image into a complete, walkable 3D virtual reality environment. The project is an implementation of the paper "3D Pano Inpainting: Building a VR Environment from a Single Input Panorama" from IEEE VRW 2024. The core idea is to generate a textured 3D mesh from the panorama which can then be rendered in real-time in a VR headset, supporting 6-degrees-of-freedom (6DOF) movement.

The pipeline consists of three main stages, each housed in a distinct directory:
1.  **Depth Estimation (`depth-estimation/`)**: This is the most complex stage. It uses the **360MonoDepth** method to generate a consistent, high-resolution 360° depth map from the input panorama. This process is not a simple inference pass. Instead, it projects the equirectangular panorama onto the 20 faces of an icosahedron, creating 20 perspective tangent images. A state-of-the-art monocular depth estimation model (e.g., MiDaS, or in this fork's focus, DepthAnything) is run on each of these 20 images. The resulting depth maps are then stitched back together using a sophisticated C++ optimization routine (exposed to Python via pybind11) that ensures global consistency and minimizes seams. This stage heavily relies on the `BoostingMonocularDepth` technique to enhance detail by merging estimations from different resolutions. The entire depth estimation process is containerized with a `Dockerfile` for easy dependency management.
2.  **Meshing and Inpainting (`inpainting/`)**: This stage takes the generated 360° depth map and the original color panorama to create a 3D mesh. It is based on the "3D Photography using Context-aware Layered Depth Inpainting" project. A Layered Depth Image (LDI) is created, and occluded regions (areas not visible in the original panorama but which become visible when moving in 3D) are "hallucinated" or filled in using a learned inpainting model. This model synthesizes both color and depth for the occluded regions, creating a more complete and immersive 3D scene. This stage also has its own `Dockerfile`.
3.  **Mesh Post-processing and Visualization (`mesh/`, `docs/`)**: After the inpainted mesh is generated (as a `.glb` file), scripts in the `mesh/` directory can be used for final adjustments, such as scaling the mesh based on a known camera height. The `docs/` directory contains a web-based viewer built with three.js that allows for interactive, real-time exploration of the final 3D meshes in a browser, with support for VR headsets.

The project is orchestrated via shell scripts (`run_360monodepth.sh`, `run_3d_photo_inpainting.sh`) which build and run the Docker containers for the respective stages, demonstrating a clear separation of concerns in the pipeline.

### Key Code and Structure Details

1.  **Pipeline Orchestration (`run_360monodepth.sh`, `run_3d_photo_inpainting.sh`)**: The two main stages of the project are executed via shell scripts. These scripts first build a Docker image for the stage (`docker build ...`) and then run a container from that image (`docker run ...`). This encapsulates all complex dependencies (C++ libraries like Ceres, Eigen, and Python libraries like PyTorch, Transformers) and makes the project highly reproducible. The `docker run` commands mount the local `data/` and `results/` directories into the container, allowing the containerized process to read inputs and write outputs directly to the host filesystem.

2.  **360-Degree Depth Estimation (`depth-estimation/360monodepth/`)**: This is the most intricate part of the repository. It's not a single model but a multi-step process for generating a globally consistent panoramic depth map.
    *   **Icosahedron Projection (`utility/projection_icosahedron.py`)**: The `erp2ico_image` function takes the input equirectangular panorama and projects it onto 20 tangent planes, each corresponding to a face of an icosahedron. This creates 20 standard perspective-projection images that can be fed into monocular depth estimation models.
    *   **Monocular Depth Estimation (`utility/depthmap_utils.py`)**: The `run_persp_monodepth` function acts as a dispatcher. Based on a command-line argument (`--persp_monodepth`), it calls the appropriate function to perform depth estimation on the 20 tangent images. The focus of this project was to add a `DepthAnything` option here.
    *   **C++ Stitching Backend (`code/cpp/`)**: The core of ensuring the 20 individual depth maps form a coherent whole is handled by a C++ backend. It implements a global optimization problem to align the depth maps. The key files are `depthmap_stitcher.hpp` and `depthmap_stitcher_group.hpp`. This optimization minimizes reprojection errors between overlapping regions of the tangent images.
    *   **Python/C++ Binding (`code/cpp/python/`, `code/cpp/src/python_binding.cpp`)**: The C++ optimization code is made available to the main Python script via `pybind11`. The `setup.py` file compiles and links the C++ code into a Python module (`instaOmniDepth`), allowing Python to call the complex alignment functions directly.

3.  **Context-Aware Inpainting (`inpainting/`)**: This directory is a fork of the "3D Photography" project. The file `inpainting/mesh.py` is central. The `write_ply` function orchestrates the inpainting process. It identifies occluded regions by creating a Layered Depth Image (LDI) and then uses pretrained models (`depth_edge_model`, `depth_feat_model`, `rgb_model`) to fill in the missing color and depth information. This is what allows for the 6DOF "walkable" experience, as it synthesizes geometry and texture for areas that were hidden in the original static image. The option `use_stable_diffusion: True` in `argument.yml` suggests an alternative inpainting backend using Stable Diffusion.

4.  **Web Viewer (`docs/renderer-uv.html`)**: The repository includes a ready-to-use web viewer built with three.js. It loads a `.glb` file specified in the URL parameters and sets up a scene with OrbitControls for desktop navigation and VRButton for VR headset integration. This allows for immediate visualization and interaction with the generated 3D content without needing specialized software.

### Focus Summary

The primary focus of the work was to replace the original monocular depth estimation model within the `360MonoDepth` pipeline with the state-of-the-art **DepthAnything** model. This is a crucial component, as the quality of the initial depth estimation directly impacts the final 3D mesh quality.

**Purpose and Importance**

The original `360MonoDepth` pipeline was designed to work with models like MiDaS. By integrating DepthAnything, the project leverages a more powerful and recent model known for its high-quality, zero-shot depth estimation across a wide variety of scenes. This is particularly important for a tool intended to work on arbitrary user-provided panoramas, which can feature diverse indoor and outdoor environments. Making the depth estimation model a pluggable component (`--persp_monodepth` argument) also improves the project's architecture, allowing for easier experimentation and future upgrades with newer models.

**How It Works and Code Examples**

The integration is primarily implemented in `depth-estimation/360monodepth/code/python/src/utility/depthmap_utils.py`.

1.  **Model Dispatching**: The `run_persp_monodepth` function acts as a factory, selecting the depth estimation function based on the `persp_monodepth` string argument. The addition of `depthanything` and `depthanythingv2` choices routes the execution to the new functions.

    ```python
    # in depth-estimation/360monodepth/code/python/src/utility/depthmap_utils.py
    def run_persp_monodepth(rgb_image_data_list, persp_monodepth, use_large_model=True):
        if (persp_monodepth == "midas2") or (persp_monodepth == "midas3"):
            return MiDaS_torch_hub_data(...)
        if persp_monodepth == "boost":
            return boosting_monodepth(...)
        if persp_monodepth == "depthanything":
            return DepthAnything(rgb_image_data_list)
        if persp_monodepth == "depthanythingv2":
            return DepthAnythingV2(rgb_image_data_list)
    ```

2.  **DepthAnythingV2 Implementation**: The new `DepthAnythingV2` function leverages the `transformers` library from Hugging Face to easily load and run the pretrained model. The function iterates through the list of 20 tangent images (`rgb_image_data_list`) generated from the icosahedron projection.

    Here's a breakdown of its implementation:
    *   **Model Loading**: It uses `AutoImageProcessor` and `AutoModelForDepthEstimation` to load the model and its corresponding pre-processor from the Hugging Face Hub. This abstracts away the model's specific architecture.
        ```python
        # in depth-estimation/360monodepth/code/python/src/utility/depthmap_utils.py
        image_processor = AutoImageProcessor.from_pretrained("pcuenq/Depth-Anything-V2-Large-hf")
        model = AutoModelForDepthEstimation.from_pretrained("pcuenq/Depth-Anything-V2-Large-hf")
        model = model.to(device)
        model.eval()
        ```
    *   **Inference Loop**: For each of the 20 tangent images, it preprocesses the image, runs inference through the model, and then resizes the output depth map to match the original tangent image's dimensions.
        ```python
        # in depth-estimation/360monodepth/code/python/src/utility/depthmap_utils.py
        disparity_map_list = []
        for index, image in enumerate(rgb_image_data_list):
            # prepare image for the model
            inputs = image_processor(images=image, return_tensors="pt")
            inputs = {name: tensor.to(device) for name, tensor in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)
                predicted_depth = outputs.predicted_depth

            # interpolate to original size
            prediction = torch.nn.functional.interpolate(
                predicted_depth.unsqueeze(1),
                size=image.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze().cpu().numpy()

            disparity_map_list.append(prediction)
        ```
    The function returns `disparity_map_list`, a list containing the 20 individual depth maps, which are then passed to the C++ backend for stitching.

**Integration into Pipeline**

The `DepthAnything` model is invoked from the main execution script `depth-estimation/360monodepth/code/python/src/main.py`. The `--persp_monodepth` command-line argument, which defaults to `depthanything`, controls which model is used. The `depthmap_estimation` function within this script calls `run_persp_monodepth`, passing along this choice, thereby integrating the new model seamlessly into the existing `360MonoDepth` pipeline. This modular design is a key strength, allowing for easy substitution of the core depth estimation component.
