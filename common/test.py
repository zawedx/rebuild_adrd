from ml_frame import with_local_info, ml_frame

def set_epoch_func():
    ml_frame._local_info["epoch"][0] = 5


@with_local_info
def test_func(epoch=None):
    print(epoch)
    set_epoch_func()
    print(epoch)

# for epoch in range(10):
#     ml_frame.set_local_info('epoch', epoch)
#     print(epoch)
#     test_func()
    
import toml
config = toml.load('config.toml')
print(config)

ml_frame.set_local_info('epoch', [1, 2, 3, 7, 7, 7])
test_func()
test_func()