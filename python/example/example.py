from libsvm.svmutil import *
y, x = svm_read_problem('heart_scale')
m = svm_train(y[:200], x[:200], '-s 5')
p_label, p_acc, p_val = svm_predict(y[200:], x[200:], m)