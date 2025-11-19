"""Test channel parsing for AuTa_0006.sxm"""

from stm_toolkit import SXMFile

sxm = SXMFile(r'C:\Users\hnhua\Documents\Code\tests\AuTa_0006.sxm')
sxm.load()

print('Channels and data shapes:')
for ch in sxm.get_channel_names():
    print(f'  {ch}:')
    print(f'    Forward: {sxm.data[ch][0].shape}, range: [{sxm.data[ch][0].min():.3e}, {sxm.data[ch][0].max():.3e}]')
    print(f'    Backward: {sxm.data[ch][1].shape}, range: [{sxm.data[ch][1].min():.3e}, {sxm.data[ch][1].max():.3e}]')

print(f'\nExpected: Z (m), Current (A), LI_Demod_1_X (A), LI_Demod_1_Y (A)')
print(f'Found: {sxm.get_channel_names()}')

