# Internal Information


## Evaluation

Evaluation of the trained model can be performed in two manners:
1. one-shot evaluation <br>
   In this evaluation, the model predicts paths based on a single depth and semantic measurement. Then quality of the paths are evaluated. Paths are classified as successful if the obstalce loss along the never exceeds a threshold (currently 0.3). Script are provided for:
   - [real-world data](./eval/eval_real_static.py)
   - [simulation data](./eval/eval_sim_static.py)
2. sequential evaluation <br>
   In this evaluation, whole paths are executed with the robot. In comparison to the one-shot evaluation, the robot dimensions are therefore taken into account. Paths are considered successful if the goal is reached within a threshold. In addition, base and knee collision rates are recoded. Script are provided for:
   - [real-world data](./eval/eval_real_dynamic.py)
   - [simulation within Nvidia Isaac Sim](https://github.com/leggedrobotics/orbit/blob/dev/pascal/anymal-vip/source/extensions/omni.isaac.anymal/omni/isaac/anymal/viplanner/evaluator.py)

REMARK: for both real-world evaluation, the data has first to be extracted from the rosbag using the script `viplanner/utils/rosbag_extractor.py`
