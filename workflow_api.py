import os
import random
import sys
from typing import Sequence, Mapping, Any, Union
import torch


def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    """Returns the value at the given index of a sequence or mapping.

    If the object is a sequence (like list or string), returns the value at the given index.
    If the object is a mapping (like a dictionary), returns the value at the index-th key.

    Some return a dictionary, in these cases, we look for the "results" key

    Args:
        obj (Union[Sequence, Mapping]): The object to retrieve the value from.
        index (int): The index of the value to retrieve.

    Returns:
        Any: The value at the given index.

    Raises:
        IndexError: If the index is out of bounds for the object and the object is not a mapping.
    """
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]


def find_path(name: str, path: str = None) -> str:
    """
    Recursively looks at parent folders starting from the given path until it finds the given name.
    Returns the path as a Path object if found, or None otherwise.
    """
    # If no path is given, use the current working directory
    if path is None:
        path = os.getcwd()

    # Check if the current directory contains the name
    if name in os.listdir(path):
        path_name = os.path.join(path, name)
        print(f"{name} found: {path_name}")
        return path_name

    # Get the parent directory
    parent_directory = os.path.dirname(path)

    # If the parent directory is the same as the current directory, we've reached the root and stop the search
    if parent_directory == path:
        return None

    # Recursively call the function with the parent directory
    return find_path(name, parent_directory)


def add_comfyui_directory_to_sys_path() -> None:
    """
    Add 'ComfyUI' to the sys.path
    """
    comfyui_path = find_path("ComfyUI")
    if comfyui_path is not None and os.path.isdir(comfyui_path):
        sys.path.append(comfyui_path)
        print(f"'{comfyui_path}' added to sys.path")


def add_extra_model_paths() -> None:
    """
    Parse the optional extra_model_paths.yaml file and add the parsed paths to the sys.path.
    """
    try:
        from main import load_extra_path_config
    except ImportError:
        print(
            "Could not import load_extra_path_config from main.py. Looking in utils.extra_config instead."
        )
        from utils.extra_config import load_extra_path_config

    extra_model_paths = find_path("extra_model_paths.yaml")

    if extra_model_paths is not None:
        load_extra_path_config(extra_model_paths)
    else:
        print("Could not find the extra_model_paths config file.")


add_comfyui_directory_to_sys_path()
add_extra_model_paths()


def import_custom_nodes() -> None:
    """Find all custom nodes in the custom_nodes folder and add those node objects to NODE_CLASS_MAPPINGS

    This function sets up a new asyncio event loop, initializes the PromptServer,
    creates a PromptQueue, and initializes the custom nodes.
    """
    import asyncio
    import execution
    from nodes import init_extra_nodes
    import server

    # Creating a new event loop and setting it as the default loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Creating an instance of PromptServer with the loop
    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)

    # Initializing custom nodes
    init_extra_nodes()


from nodes import (
    NODE_CLASS_MAPPINGS,
    VAELoader,
    DualCLIPLoader,
    LoraLoaderModelOnly,
    EmptyLatentImage,
    SaveImage,
    LoadImage,
    CheckpointLoaderSimple,
    VAEDecode,
)


def main():
    import_custom_nodes()
    with torch.inference_mode():
        vaeloader = VAELoader()
        vaeloader_195 = vaeloader.load_vae(vae_name="flux/ae.safetensors")

        dualcliploader = DualCLIPLoader()
        dualcliploader_224 = dualcliploader.load_clip(
            clip_name1="EVA02_CLIP_L_336_psz14_s6B.1.pt",
            clip_name2="sd3m/t5xxl_fp8_e4m3fn.safetensors",
            type="flux",
            device="default",
        )

        ksamplerselect = NODE_CLASS_MAPPINGS["KSamplerSelect"]()
        ksamplerselect_293 = ksamplerselect.get_sampler(sampler_name="euler")

        randomnoise = NODE_CLASS_MAPPINGS["RandomNoise"]()
        randomnoise_295 = randomnoise.get_noise(noise_seed=random.randint(1, 2**64))

        pulidfluxmodelloader = NODE_CLASS_MAPPINGS["PulidFluxModelLoader"]()
        pulidfluxmodelloader_365 = pulidfluxmodelloader.load_model(
            pulid_file="pulid_flux_v0.9.0.safetensors"
        )

        pulidfluxevacliploader = NODE_CLASS_MAPPINGS["PulidFluxEvaClipLoader"]()
        pulidfluxevacliploader_366 = pulidfluxevacliploader.load_eva_clip()

        pulidfluxinsightfaceloader = NODE_CLASS_MAPPINGS["PulidFluxInsightFaceLoader"]()
        pulidfluxinsightfaceloader_367 = pulidfluxinsightfaceloader.load_insightface(
            provider="CUDA"
        )

        fluxresolutionnode = NODE_CLASS_MAPPINGS["FluxResolutionNode"]()
        fluxresolutionnode_390 = fluxresolutionnode.calculate_dimensions(
            megapixel="1.0",
            aspect_ratio="1:1 (Perfect Square)",
            divisible_by="8",
            custom_ratio="1:1",
            custom_aspect_ratio="1:1",
        )

        checkpointloadersimple = CheckpointLoaderSimple()
        checkpointloadersimple_438 = checkpointloadersimple.load_checkpoint(
            ckpt_name="robux-machine.safetensors"
        )

        loraloadermodelonly = LoraLoaderModelOnly()
        loraloadermodelonly_429 = loraloadermodelonly.load_lora_model_only(
            lora_name="flux/lora.safetensors",
            strength_model=0.8,
            model=get_value_at_index(checkpointloadersimple_438, 0),
        )

        loadimage = LoadImage()
        loadimage_450 = loadimage.load_image(image="input.png")

        cachingcliptextencode = NODE_CLASS_MAPPINGS["CachingCLIPTextEncode"]()
        cachingcliptextencode_451 = cachingcliptextencode.encode(
            text="deformed, dark, unrealistic, plastic-like skin, blurry suit, unfocused suit, blurry tie, blurry edges, asymmetrical, angry, zoomed-in, too-close, facing away, blurred body, unfocused, blurry face, ears sticking out, deformed ears, small-sized body, long neck, blurred background, animated appearance, fake appearance, small body appearance, thin body, scrawny appearance, unconfident, tiny and thin shoulders, hands, fingers",
            clip=get_value_at_index(dualcliploader_224, 0),
        )

        cachingcliptextencode_452 = cachingcliptextencode.encode(
            text="Professional photorealistic LinkedIn portrait of a {sex}; faceswapped; identical hair (style & color) and original facial structure preserved; confident, approachable expression; impeccably tailored business suit & perfectly knotted plain-color tie with visible fine fabric fibers; medium portrait captured from several meters away (subject occupies roughly one-third of frame height, background more pronounced), head-to-upper-torso visible, no distracting hands foregrounded; shot with 135 mm f/2 full-frame DSLR, ISO 100, creamy bokeh isolating subject; balanced natural daylight key with soft fill and gentle rim light; ultra-high-resolution, crisp facial detail despite distance; no text or watermark.\n\nsex = male/man or female/woman\n\nBackground variants (choose one per render):\n• Sunlit urban park path with lush green bokeh balls.\n• Spacious glass atrium or sky-bridge flooded with ambient light.\n• Grand corporate lobby with diffused skylight, marble accents.\n• Bright, airy boardroom seen through wide windows.\n\nRule-of-thirds eye-line or centered symmetry; polished, cinematic color grade; sharp yet natural finish.\n\nrealistic body-to-head ratio",
            clip=get_value_at_index(dualcliploader_224, 0),
        )

        loraloadermodelonly_453 = loraloadermodelonly.load_lora_model_only(
            lora_name="flux/flux.1-turbo-alpha/diffusion_pytorch_model.safetensors",
            strength_model=0.7000000000000001,
            model=get_value_at_index(loraloadermodelonly_429, 0),
        )

        emptylatentimage = EmptyLatentImage()
        applypulidflux = NODE_CLASS_MAPPINGS["ApplyPulidFlux"]()
        fluxforwardoverrider = NODE_CLASS_MAPPINGS["FluxForwardOverrider"]()
        applyteacachepatch = NODE_CLASS_MAPPINGS["ApplyTeaCachePatch"]()
        fluxguidance = NODE_CLASS_MAPPINGS["FluxGuidance"]()
        cfgguider = NODE_CLASS_MAPPINGS["CFGGuider"]()
        modelsamplingflux = NODE_CLASS_MAPPINGS["ModelSamplingFlux"]()
        basicscheduler = NODE_CLASS_MAPPINGS["BasicScheduler"]()
        samplercustomadvanced = NODE_CLASS_MAPPINGS["SamplerCustomAdvanced"]()
        vaedecode = VAEDecode()
        saveimage = SaveImage()

        for q in range(1):
            emptylatentimage_5 = emptylatentimage.generate(
                width=get_value_at_index(fluxresolutionnode_390, 0),
                height=get_value_at_index(fluxresolutionnode_390, 1),
                batch_size=1,
            )

            applypulidflux_364 = applypulidflux.apply_pulid_flux(
                weight=0.9,
                start_at=0.1,
                end_at=1,
                model=get_value_at_index(loraloadermodelonly_453, 0),
                pulid_flux=get_value_at_index(pulidfluxmodelloader_365, 0),
                eva_clip=get_value_at_index(pulidfluxevacliploader_366, 0),
                face_analysis=get_value_at_index(pulidfluxinsightfaceloader_367, 0),
                image=get_value_at_index(loadimage_450, 0),
                unique_id=8181723305799803326,
            )

            fluxforwardoverrider_441 = fluxforwardoverrider.apply_patch(
                model=get_value_at_index(applypulidflux_364, 0)
            )

            applyteacachepatch_446 = applyteacachepatch.apply_patch(
                rel_l1_thresh=0.4,
                cache_device="offload_device",
                wan_coefficients="disabled",
                model=get_value_at_index(fluxforwardoverrider_441, 0),
            )

            fluxguidance_300 = fluxguidance.append(
                guidance=2.5,
                conditioning=get_value_at_index(cachingcliptextencode_452, 0),
            )

            fluxguidance_447 = fluxguidance.append(
                guidance=2.5,
                conditioning=get_value_at_index(cachingcliptextencode_451, 0),
            )

            cfgguider_298 = cfgguider.get_guider(
                cfg=1,
                model=get_value_at_index(applyteacachepatch_446, 0),
                positive=get_value_at_index(fluxguidance_300, 0),
                negative=get_value_at_index(fluxguidance_447, 0),
            )

            modelsamplingflux_402 = modelsamplingflux.patch(
                max_shift=1,
                base_shift=0.5,
                width=get_value_at_index(fluxresolutionnode_390, 0),
                height=get_value_at_index(fluxresolutionnode_390, 1),
                model=get_value_at_index(loraloadermodelonly_453, 0),
            )

            basicscheduler_294 = basicscheduler.get_sigmas(
                scheduler="simple",
                steps=10,
                denoise=1,
                model=get_value_at_index(modelsamplingflux_402, 0),
            )

            samplercustomadvanced_292 = samplercustomadvanced.sample(
                noise=get_value_at_index(randomnoise_295, 0),
                guider=get_value_at_index(cfgguider_298, 0),
                sampler=get_value_at_index(ksamplerselect_293, 0),
                sigmas=get_value_at_index(basicscheduler_294, 0),
                latent_image=get_value_at_index(emptylatentimage_5, 0),
            )

            vaedecode_29 = vaedecode.decode(
                samples=get_value_at_index(samplercustomadvanced_292, 0),
                vae=get_value_at_index(vaeloader_195, 0),
            )

            saveimage_30 = saveimage.save_images(
                filename_prefix="Flux/Flux", images=get_value_at_index(vaedecode_29, 0)
            )


if __name__ == "__main__":
    main()
