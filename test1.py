import mnist_loader
import network
import pic_to_testdata


directory_input = 'mydata_input'
directory_input = 'mydata_input_1'
directory_output = 'mydata_output'
pkl_path_test = 'mydata.pkl.gz'
images_data, array_data, label_data,array_data_out, label_data_out = pic_to_testdata.convert_images_to_mnist_format(directory_input)
print((array_data_out))
pic_to_testdata.save_as_pkl_gz([array_data_out, label_data_out], pkl_path_test)
pic_to_testdata.save_images(images_data, label_data, directory_output)

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
net = network.Network([784, 30, 10])
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
 
# net = network.load_net('mynet/mynet_94.4')
# net.evaluate(test_data)





