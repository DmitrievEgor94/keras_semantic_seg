import pickle
from matplotlib import pyplot as pl

history_gorot = pickle.load(open('history/Moscow/LinkNet_history_glorot_uniform', 'rb'))
history_pretrained_imagenet = pickle.load(open('history/Moscow/LinkNet_history_pretrained', 'rb'))
history_pretrained_on_Tatarstan = pickle.load(open('history/Moscow/LinkNet_history_pretrained_from_Tatarstan', 'rb'))
x_l = list(range(0, 30))

print(history_gorot.keys())

#ключи: val_acc, loss, acc, val_loss, lr

y_l = history_gorot['acc']
y_u = history_pretrained_imagenet['acc']
y_r = history_pretrained_on_Tatarstan['acc']
pl.title('Точность сегментации на тренировачной выборке')

pl.plot(x_l,y_l, label='Распределение Ксавьера')
pl.plot(x_l, y_u, label='Предобученный декодер')
pl.plot(x_l, y_r, label='Предобучение на Татарстане')

pl.legend(loc=7)
pl.show()

print(max(y_l))
print(max(y_r))
print(max(y_u))


# for key in history.keys():
#     print()key