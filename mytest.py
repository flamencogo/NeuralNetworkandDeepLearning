import mnist_loader
import network
import pic_to_testdata


directory_input = 'mydata_input'
# directory_input = 'mydata_input_1'
directory_output = 'mydata_output'
pkl_path_mydata = 'mydata.pkl.gz'
pkl_path_mydata_train = 'mydata_train.pkl.gz'
pkl_path_mydata_test = 'mydata_test.pkl.gz'
images_data, array_data, label_data,array_data_out, label_data_out = pic_to_testdata.convert_images_to_mnist_format(directory_input)
mydata_train,mydata_train_y,mydata_test,mydata_test_y = pic_to_testdata.array_split(array_data_out,label_data_out,0)
print(len(mydata_train)) # 84
pic_to_testdata.save_as_pkl_gz([array_data_out, label_data_out], pkl_path_mydata)
pic_to_testdata.save_as_pkl_gz([mydata_train, mydata_train_y], pkl_path_mydata_train)
pic_to_testdata.save_as_pkl_gz([mydata_test, mydata_test_y], pkl_path_mydata_test)
pic_to_testdata.save_images(images_data, label_data, directory_output)

training_data, validation_data, test_data, mydata_train_data = mnist_loader.load_data_wrapper()
net = network.Network([784,30,10])
net.SGD(training_data, mydata_train_data, 50, 10, 3.0, test_data=test_data)
 
# net = network.load_net('mynet/mynet_94.4')
# net.evaluate(test_data)





