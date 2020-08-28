from fastreid.modeling.meta_arch.mgn import MGN
from fastreid.config import get_cfg
from fastreid.engine import default_argument_parser, default_setup


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


args = default_argument_parser().parse_args()
cfg = get_cfg()
mm = MGN(cfg)
print(mm)
