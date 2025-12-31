from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.evalutaion import Evaluation
from cnnClassifier import logger

STAGE_NAME = "Evaluation Stage"
class EvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        val_config = config.get_validation_config()
        evaluat = Evaluation(val_config)
        evaluat.evaluation()
        evaluat.save_score()

if __name__ == "__main__":
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        evaluation_pipeline = EvaluationPipeline()
        evaluation_pipeline.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e