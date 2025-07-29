import hailo_platform as hp
import numpy as np

hef_path = "models/rasen.hef"
hef = hp.HEF(hef_path)
vdev = hp.VDevice()
cfg = hp.ConfigureParams.create_from_hef(hef, interface=hp.HailoStreamInterface.PCIe)
ng = vdev.configure(hef, cfg)[0]

ivs_info = hef.get_input_vstream_infos()[0]
print("Input stream name:", ivs_info.name, "shape:", ivs_info.shape, "size:", ivs_info.shape)

act_params = ng.create_params()

# 入力データ作成
tensor = np.zeros(ivs_info.shape, dtype=np.uint8).ravel()
print("Tensor shape:", tensor.shape, tensor.dtype, tensor.nbytes)

in_params  = hp.InputVStreamParams.make_from_network_group(ng, quantized=True, format_type=hp.FormatType.UINT8)
out_params = hp.OutputVStreamParams.make_from_network_group(ng, quantized=True, format_type=hp.FormatType.UINT8)
_in_holder  = hp.InputVStreams(ng, in_params).__enter__()
_out_holder = hp.OutputVStreams(ng, out_params).__enter__()
ivs = next(iter(_in_holder))
ovs = next(iter(_out_holder))

with ng.activate(act_params):
    ivs.send(tensor)
    out_buf = np.empty(2, dtype=np.uint8)
    ovs.recv(out_buf)
print("SUCCESS, out_buf:", out_buf)
