from src.policy.baseline.bc_baseline import Simple_driver
from src.policy.latentdriver.lantentdriver_model import LantentDriver
from src.policy.easychauffeur.easychauffeur_ppo import EasychauffeurPolicy
from src.policy.latentdriver.carplan_model import CarPLAN
# from src.policy.carplan.carplan_model import PlanningModel_MoE
__all__ = {
    'baseline': Simple_driver,
    'latentdriver': LantentDriver,
    'easychauffeur': EasychauffeurPolicy,
    'carplan': CarPLAN
}


def build_model(config):
    model = __all__[config.method.model_name](**config.method)
    return model