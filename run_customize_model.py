from recbole.quick_start import run

if __name__ == "__main__":
    args = dict(
        model = 'FECA',
        dataset = 'hdf-405',
        config_file_list = ['config/model_config.yaml'],
        ip = "localhost",
        port = '5678',
        world_size = -1,
        nproc = 1,
        group_offset = 0
    )

    run(
        args['model'],
        args['dataset'],
        args['config_file_list'],
        nproc=args['nproc'],
        world_size=args['world_size'],
        ip=args['ip'],
        port=args['port'],
        group_offset=args['group_offset'],
    )