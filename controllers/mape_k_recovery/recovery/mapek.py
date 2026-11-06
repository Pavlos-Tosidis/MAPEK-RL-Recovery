"""MAPE-K monitoring and decision state for targeted recovery."""
from __future__ import annotations
from dataclasses import dataclass,field

@dataclass
class RecoveryKnowledge:
    base_model: str
    active_model: str
    onset: int=5
    cooldown_steps: int=100
    current_state: str="nominal"
    anomaly_counts: dict[str,int]=field(default_factory=lambda:{"image":0,"lidar":0})
    cooldown: dict[str,int]=field(default_factory=lambda:{"image":0,"lidar":0})
    diagnosed: dict=field(default_factory=dict)
    recovered_modalities: set[str]=field(default_factory=set)
    adaptation_count: int=0
    history: list[dict]=field(default_factory=list)

    def update(self,evidence:dict)->set[str]:
        """Consecutive learned-detector positives trigger active diagnosis."""
        matched=set()
        for modality in ("image","lidar"):
            if self.cooldown[modality]:
                self.cooldown[modality]-=1
                self.anomaly_counts[modality]=0
                continue
            flag=bool(evidence[f"{modality}_raw"])
            self.anomaly_counts[modality]=self.anomaly_counts[modality]+1 if flag else 0
            if self.anomaly_counts[modality]>=self.onset and modality not in self.recovered_modalities:
                matched.add(modality)
        return matched

    def reject(self, modalities:set[str], errors:dict):
        for name in modalities:
            self.anomaly_counts[name]=0
            self.cooldown[name]=self.cooldown_steps
        self.history.append({"event":"diagnosis_rejected","modalities":sorted(modalities),"errors":errors})

    def redeployed(self,descriptors,model_path):
        self.current_state="recovered"
        self.active_model=str(model_path)
        self.diagnosed.update(descriptors)
        self.recovered_modalities.update(descriptors)
        self.adaptation_count+=1
        self.history.append({"event":"recovery_deployed","descriptors":descriptors,
                             "model":str(model_path)})

    def reset_episode(self):
        self.anomaly_counts={"image":0,"lidar":0}
        self.cooldown={"image":0,"lidar":0}
