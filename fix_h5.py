import h5py

path = 'agrimater_model2.h5'
f = h5py.File(path, 'r')
print('Root attrs:', list(f.attrs.keys()))
for obj in f.values():
    print(obj.name, list(obj.attrs.keys()))
f.close()
