from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.training import Training
from cnnClassifier.components.prepare_callbacks import PrepareCallback
from cnnClassifier import logger

STAGE_NAME = "Training Stage"

class TrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        prepare_callbacks_config = config.get_prepare_callback_config()
        prepare_callbacks = PrepareCallback(config=prepare_callbacks_config)
        callback_list = prepare_callbacks.get_tb_ckpt_callbacks()

        training_config = config.get_training_config()
        training = Training(config=training_config)
        training.get_base_model()
        training.train_valid_generator()
        training.train(
            callback_list=callback_list
        )

if __name__== "main":
    try:
        logger.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<")
        training_pipeline = TrainingPipeline()
        training_pipeline.main()
        logger.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e