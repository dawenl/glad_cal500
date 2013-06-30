# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import glob, operator
import numpy as np

import glad_em

# <codecell>

%cd annotations_usr/
files = glob.glob('*.mp3.txt')
%cd ../

# <codecell>

annotator2song, annotator2id = {}, {}
song2annotator, song2id = {}, {}
annotator_count, song_count = 0, 0
for f in files:
    tmp = f.strip().split('-', 1)
    if tmp[0] not in annotator2song.keys():
        annotator2song[tmp[0]] = [tmp[1]]
        annotator2id[tmp[0]] = annotator_count
        annotator_count += 1
    else:
        annotator2song[tmp[0]].append(tmp[1])
        
    if tmp[1] not in song2annotator.keys():
        song2annotator[tmp[1]] = [tmp[0]]
        song2id[tmp[1]] = song_count
        song_count += 1
    else:
        song2annotator[tmp[1]].append(tmp[0])

# <codecell>

song_per_annotator = [len(songs) for songs in annotator2song.values()]
annotator_per_song = [len(annotators) for annotators in song2annotator.values()]

# <codecell>

figure()
hist(song_per_annotator, bins=50)
figure()
hist(annotator_per_song, bins=15)
pass

# <codecell>

def extract_label(label, values_p, values_n=None):
    if values_n is None:
        Labels = np.zeros((annotator_count, song_count))
    else:
        Labels = -np.ones((annotator_count, song_count))
    for f in files:
        annotator, song = f.strip().split('-', 1)
        data = np.loadtxt('annotations_usr/' + f, dtype='str', delimiter=' = ', skiprows=3)
        for (k, v) in data:
            if k == label:
                v = v.lower()
                if v in values_p:
                    Labels[annotator2id[annotator], song2id[song]] = 1
                elif values_n is not None and v in values_n:
                    Labels[annotator2id[annotator], song2id[song]] = 0
                break
    return Labels 

def show_values(label):
    values = []
    for f in files:
        data = np.loadtxt('annotations_usr/' + f, dtype='str', delimiter=' = ', skiprows=3)
        for (k, v) in data:
            if k == label:
                v = v.lower()
                if v not in values:
                    values.append(v)
                break
    return values

def output_info(aver_acc):
    best = -inf
    worst = inf
    none = inf
    for (k, v) in aver_acc.items():
        if best < v:
            best = v
            blabel = k
        if worst > v:
            worst = v
            wlabel= k
        if none > abs(v):
            none = v
            nlabel = k
    
    print 'Best: {}: {}'.format(blabel, best)
    print 'Worst: {}: {}'.format(wlabel, worst)
    print 'None: {}: {}'.format(nlabel, none)
    pass

sorted_x = sorted(x.iteritems(), key=operator.itemgetter(1))   
sorted_x

# <codecell>

id2song = {}
for (k, v) in song2id.items():
    id2song[v] = k

id2annotator = {}
for (k, v) in annotator2id.items():
    id2annotator[v] = k

# <codecell>

label = 'Genre-Blues'
show_values(label)

# <codecell>

labels = np.loadtxt('vocab.txt', dtype='str')

# <codecell>

## Instrument level

reload(glad_em)

missing = True
inst_aver_acc = {}
for label in labels:
    if label.startswith('Instrument') and not label.endswith('Solo'):
        Labels = extract_label(label, ['"present"', '"prominent"'], values_n=['"none"'])
        if sum(Labels == -1) == Labels.size:
            print '{} has no annotation'.format(label)
            continue
        glad = glad_em.GLAD(Labels)
        maxiter = 10
        threshold = 0.001
        old_obj = -np.inf
        for i in xrange(maxiter):
            glad.e_step()
            glad.m_step(missing=missing)
            improvement = (glad.obj - old_obj) / abs(glad.obj)
            print 'After ITERATION: {}\tObjective: {:.2f}\tOld objective: {:.2f}\tImprovement: {:.4f}'.format(i, glad.obj, old_obj, improvement)
            if improvement < threshold:
                break
            old_obj = glad.obj
        
        print 'For label {}:'.format(label)
        print '\tAverage user accuracy is {}'.format(mean(glad.alpha))
        print '\tThe most expert user is {}'.format(id2annotator[argmax(glad.alpha)])
        print '\tThe lest expert user is {}'.format(id2annotator[argmin(glad.alpha)])
        print '\tThe hardest song is {}, the easiest song is {}'.format(id2song[argmin(glad.beta)], id2song[argmax(glad.beta)])
        print
        inst_aver_acc[label] = mean(glad.alpha)
output_info(inst_aver_acc)
sorted_inst = sorted(inst_aver_acc.iteritems(), key=operator.itemgetter(1))           

# <codecell>

sorted_inst

# <codecell>

## solo
missing = False
solo_aver_acc = {}
for label in labels:
    if label.startswith('Instrument') and label.endswith('Solo'):
        Labels = extract_label(label, ['"yes"'])
        if sum(Labels == -1) == Labels.size:
            print '{} has no annotation'.format(label)
            continue
        glad = glad_em.GLAD(Labels)
        maxiter = 10
        threshold = 0.001
        old_obj = -np.inf
        for i in xrange(maxiter):
            glad.e_step()
            glad.m_step(missing=missing)
            improvement = (glad.obj - old_obj) / abs(glad.obj)
            print 'After ITERATION: {}\tObjective: {:.2f}\tOld objective: {:.2f}\tImprovement: {:.4f}'.format(i, glad.obj, old_obj, improvement)
            if improvement < threshold:
                break
            old_obj = glad.obj
        
        print 'For label {}:'.format(label)
        print '\tAverage user accuracy is {}'.format(mean(glad.alpha))
        print '\tThe most expert user is {}'.format(id2annotator[argmax(glad.alpha)])
        print '\tThe lest expert user is {}'.format(id2annotator[argmin(glad.alpha)])
        print '\tThe hardest song is {}, the easiest song is {}'.format(id2song[argmin(glad.beta)], id2song[argmax(glad.beta)])
        print
        solo_aver_acc[label] = mean(glad.alpha)
output_info(solo_aver_acc)
sorted_solo = sorted(solo_aver_acc.iteritems(), key=operator.itemgetter(1))           

# <codecell>

sorted_solo

# <codecell>

bins = numpy.linspace(-2, 12, 50)
hist(solo_aver_acc.values(), bins, alpha=0.5)
hist(inst_aver_acc.values(), bins, alpha=0.5)
legend(["Solo", "Instrument"])
savefig('comp.png')
pass

# <codecell>

## Genre
missing = False
#genre_aver_acc = {}
for label in labels:
    if label.startswith('Genre') and not label.startswith('Genre-Best') and not label.startswith('Genre--_'):
        Labels = extract_label(label, ['"yes"'])
        if sum(Labels == -1) == Labels.size:
            print '{} has no annotation'.format(label)
            continue
        glad = glad_em.GLAD(Labels)
        maxiter = 10
        threshold = 0.001
        old_obj = -np.inf
        for i in xrange(maxiter):
            glad.e_step()
            glad.m_step(missing=missing)
            improvement = (glad.obj - old_obj) / abs(glad.obj)
            print 'After ITERATION: {}\tObjective: {:.2f}\tOld objective: {:.2f}\tImprovement: {:.4f}'.format(i, glad.obj, old_obj, improvement)
            if improvement < threshold:
                break
            old_obj = glad.obj
        
        print 'For label {}:'.format(label)
        print '\tAverage user accuracy is {}'.format(mean(glad.alpha))
        print '\tThe most expert user is {}'.format(id2annotator[argmax(glad.alpha)])
        print '\tThe lest expert user is {}'.format(id2annotator[argmin(glad.alpha)])
        print '\tThe hardest song is {}, the easiest song is {}'.format(id2song[argmin(glad.beta)], id2song[argmax(glad.beta)])
        print
        genre_aver_acc[label] = mean(glad.alpha)
output_info(genre_aver_acc)
sorted_genre = sorted(genre_aver_acc.iteritems(), key=operator.itemgetter(1))           

# <codecell>

sorted_genre

# <codecell>


