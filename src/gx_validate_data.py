from loguru import logger
import great_expectations as gx

from src.config import PROJ_ROOT
from src.gx_context_configurations import CHECKPOINT, batch_dataframes

context = gx.get_context(mode="file", project_root_dir=PROJ_ROOT)

checkpoint = context.checkpoints.get(CHECKPOINT)

checkpoint_result = checkpoint.run(
)

validation_result = checkpoint_result.run_results[list(checkpoint_result.run_results.keys())[0]]

expectations_run = validation_result["statistics"]["evaluated_expectations"]
expectations_failed = validation_result["statistics"]["unsuccessful_expectations"]

logger.info(f"Validation results: {expectations_run} expectations evaluated, {expectations_failed} expectations failed.")