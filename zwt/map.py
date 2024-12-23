from pathlib import Path


def argoverse2_load_map(instance_dir):
    log_map_dirpath = Path(instance_dir)
    from av2.map.map_api import ArgoverseStaticMap
    vector_data_fnames = sorted(log_map_dirpath.glob("log_map_archive_*.json"))
    if not len(vector_data_fnames) == 1:
        raise RuntimeError(f"JSON file containing vector map data is missing (searched in {log_map_dirpath})")
    vector_data_fname = vector_data_fnames[0]
    return ArgoverseStaticMap.from_json(vector_data_fname)


if __name__ == '__main__':
    file = r'/media/zwt/新加卷/Argoverse_2/val/00a0ec58-1fb9-4a2b-bfd7-f4e5da7a9eff/'
    scenario = argoverse2_load_map(file)
    # /media/zwt/新加卷/Argoverse_2/val/00a0ec58-1fb9-4a2b-bfd7-f4e5da7a9eff/log_map_archive_00a0ec58-1fb9-4a2b-bfd7-f4e5da7a9eff.json
    # /media/zwt/新加卷/Argoverse_2/val/00a0ec58-1fb9-4a2b-bfd7-f4e5da7a9eff/scenario_00a0ec58-1fb9-4a2b-bfd7-f4e5da7a9eff.parquet
    print('1')
    # tract_fragment(0), unscored_track(1), scored_track(2), focal_track(3), 2在多智能体轨迹预测中得分，3在单智能体轨迹预测中得分