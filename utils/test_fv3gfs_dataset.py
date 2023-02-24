from .data_loader_fv3gfs import FV3GFSDataset

TEST_PARAMS = {
    'n_history': 0,
    'crop_size_x': None,
    'crop_size_y': None,
    'roll': False,
    'in_channels': [0, 1, 4, 19],
    'out_channels': [0, 1, 4, 19],
    'two_step_training': False,
    'orography': False,
    'add_noise': False,
    'dt': 1,
    'n_history': 0,
}
TEST_PATH = '/Users/oliverwm/scratch/fv3gfs-fourcastnet/fourcastnet_vanilla_1_degree'

class DotDict:
    def __init__(self, items):
        self.items = items

    def __getitem__(self, key):
        return self.items[key]
    
    def __getattr__(self, attr):
        return self.items[attr]
    
    def __contains__(self, key):
        return (key in self.items)

def test_FV3GFSDataset_init():
    FV3GFSDataset(DotDict(TEST_PARAMS), TEST_PATH, True)