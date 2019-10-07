# def donuts(count):
#   if count < 10:
#     return 'Number of donuts: ' + str(count)
#   else:
#     return 'Number of donuts: many'
#
# print(donuts(9))
#
# x=6
# y=1
# print(x ** y)
#
# s = "\t\tWelcome\n"
# print(s.strip())
#
# class Sales:
#     def __init__(self, id):
#         self.id = id
#         id = 100
#
# val = Sales(123)
# print (val.id)
import tensorflow as tf
print(tf.__version__)

# tfe = tf.contrib # Shorthand for some symbols
#
# tf.enable_eager_execution()