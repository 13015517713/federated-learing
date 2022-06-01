import logging
from server.optimizer.fedbase import BaseServer
class Server(BaseServer):
    def __init__(self, global_model, global_trainer, global_testset, clients, options):
        super().__init__(global_model, global_trainer, global_testset, clients, options)
        self.options = options
    def run(self):
        loss_recorder, acc_recorder, _ = self.global_trainer.self_test(self.test_loader)
        logging.info("global model test, loss=%.4f, acc=%.4f."%(loss_recorder.avg, acc_recorder.avg) )
