from av2.datasets.motion_forecasting import scenario_serialization
from pathlib import Path

def argoverse2_load_scenario(instance_dir):
    """加载场景文件，返回序列化场景"""
    from av2.datasets.motion_forecasting import scenario_serialization
    file_path = sorted(Path(instance_dir).glob("*.parquet"))
    print(file_path)
    assert len(file_path) == 1
    file_path = file_path[0]
    return scenario_serialization.load_argoverse_scenario_parquet(file_path)  # 返回的type是ArgoverseScenario

if __name__ == '__main__':
    file = r'/media/zwt/新加卷/Argoverse_2/train/e0f364f3-f6ed-403d-8de0-dcf688bcd7b2/'
    scenario = argoverse2_load_scenario(file)
    # /media/zwt/新加卷/Argoverse_2/val/00a0ec58-1fb9-4a2b-bfd7-f4e5da7a9eff/log_map_archive_00a0ec58-1fb9-4a2b-bfd7-f4e5da7a9eff.json
    # /media/zwt/新加卷/Argoverse_2/val/00a0ec58-1fb9-4a2b-bfd7-f4e5da7a9eff/scenario_00a0ec58-1fb9-4a2b-bfd7-f4e5da7a9eff.parquet
    print('1')
    # tract_fragment(0), unscored_track(1), scored_track(2), focal_track(3), 2在多智能体轨迹预测中得分，3在单智能体轨迹预测中得分