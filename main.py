import torch
import hydra
from omegaconf import OmegaConf
from pipelines.pipeline import InferencePipeline
from chaplin import Chaplin


@hydra.main(version_base=None, config_path="hydra_configs", config_name="default")
def main(cfg):
    # Load before starting hotkeys/background resources, so setup errors exit cleanly.
    pipeline = InferencePipeline(
        cfg.config_filename,
        device=torch.device(
            f"cuda:{cfg.gpu_idx}" if torch.cuda.is_available() and cfg.gpu_idx >= 0 else "cpu"
        ),
        detector=cfg.detector,
        face_track=True,
        input_v_fps=16,
    )
    options = OmegaConf.to_container(cfg, resolve=True)
    if not 1 <= cfg.nbest <= pipeline.model.beam_search.beam_size:
        raise ValueError("nbest must be between 1 and the configured beam_size")
    chaplin = Chaplin(options, vsr_model=pipeline)
    print("\nMODEL LOADED SUCCESSFULLY!\n")
    chaplin.start_webcam()


if __name__ == "__main__":
    main()
