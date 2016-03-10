#!/usr/bin/env python
"""This script gives insight on Titanic survivors data."""


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.style.use('ggplot')
plt.rcParams['figure.figsize'] = (16, 8)
mpl.rcParams['figure.dpi'] = 300

# "Tableau 20" colors as RGB.
tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

# Scale RGB values to [0, 1] range.
for i in range(len(tableau20)):
    r, g, b = tableau20[i]
    tableau20[i] = (r / 255., g / 255., b / 255.)

# read data
df_titanic = sns.load_dataset('titanic')

# some insight on data
# survivalByGender
grp = df_titanic.groupby('sex')[['survived']].mean()*100
ax = grp.plot.bar(color=tableau20[0], legend=None)
ax.set_axis_bgcolor('white')
plt.xticks(np.arange(0, 5), ['Female', 'Male'], rotation=0, ha ='center', fontsize=14)
ax.set_xlabel("")
plt.yticks([])
plt.title("Survival rate by gender\n", fontsize=22, loc='left')
rects = ax.patches
labels = [np.round(value, decimals=1) for value in np.concatenate(grp.values)]
for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height-5., label, ha='center',
            va='bottom', color='white', fontsize=16)
ax.axis('tight')
plt.savefig("survivalByGender.png",
            bbox_inches='tight',
            dpi=300,
            format='png')
plt.savefig("survivalByGender.pdf",
            bbox_inches='tight',
            dpi=300,
            format='pdf')

# survivalByGenderClass
grp = df_titanic.pivot_table('survived', index='sex', columns='class')*100
ax = grp.plot.bar(color=tableau20, legend=False, align='center')
#ax.legend(loc=1)
ax.set_axis_bgcolor('white')
plt.xticks(np.arange(0, 5), ['Female', 'Male'], rotation=0, ha ='center', fontsize=14)
ax.set_xlabel("")
plt.yticks([])
plt.title("Survival rate by gender and class\n", fontsize=22, loc='left')
rects = ax.patches
labels = [np.round(value, decimals=1) for value in np.concatenate(grp.T.values)]
legends = ['First class', 'First class',
           'Second class', 'Second class',
           'Third class', 'Third class']
for rect, label, legend in zip(rects, labels, legends):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height-5., label, ha='center',
            va='bottom', color='white', fontsize=16)
    if float(label) >= 30:
        ax.text(rect.get_x() + rect.get_width()/2, height-7.5, legend,
                ha='center', va='top', color='white', fontsize=16, rotation=90)
    else:
        ax.text(rect.get_x() + rect.get_width()/2, height+7.5, legend,
                ha='center', va='bottom', color='black', fontsize=16, rotation=90)
ax.axis('tight')
plt.savefig("survivalByGenderClass.png",
            bbox_inches='tight',
            dpi=300,
            format='png')
plt.savefig("survivalByGenderClass.pdf",
            bbox_inches='tight',
            dpi=300,
            format='pdf')

# survivalByGenderClassAgeSlice
age = pd.cut(df_titanic['age'], [0, 18, 80])
grp = df_titanic.pivot_table('survived', ['sex', age], 'class').unstack()*100
ax = grp.plot.bar(align='center', color=tableau20, legend=False)
ax.set_axis_bgcolor('white')
plt.xticks(np.arange(0, 5), ['Female', 'Male'], rotation=0, ha ='center', fontsize=14)
ax.set_xlabel("")
plt.yticks([])
plt.title("Survival rate by gender and class, and age range\n", fontsize=22, loc='left')
rects = ax.patches
labels = [np.round(value, decimals=1) for value in np.concatenate(grp.T.values)]
legends = ['First class, minors', 'First class, minors',
           'First class, adults', 'First class, adults',
           'Second class, minors', 'Second class, minors',
           'Second class, adults', 'Second class, adults',
           'Third class, minors', 'Third class, minors',
           'Third class, adults', 'Third class, adults']
for rect, label, legend in zip(rects, labels, legends):
    height = rect.get_height()
    if float(label) >= 40:
        ax.text(rect.get_x() + rect.get_width()/2, height-5., label, ha='center',
                va='bottom', color='white', fontsize=16)
        ax.text(rect.get_x() + rect.get_width()/2, height-7.5, legend,
                ha='center', va='top', color='white', fontsize=16, rotation=90)
    else:
        ax.text(rect.get_x() + rect.get_width()/2,
                height-5.,
                label,
                ha='center',
                va='bottom',
                color='white',
                fontsize=16)
        ax.text(rect.get_x() + rect.get_width()/2,
                height+7.5,
                legend,
                ha='center',
                va='bottom',
                color='black',
                fontsize=16,
                rotation=90)
ax.axis('tight')
plt.savefig("survivalByGenderClassAgeSlice.png",
            bbox_inches='tight',
            dpi=300,
            format='png')
plt.savefig("survivalByGenderClassAgeSlice.pdf",
            bbox_inches='tight',
            dpi=300,
            format='pdf')


# fare = pd.qcut(df_titanic['fare'], 2)
# df_titanic.pivot_table('survived', ['sex', age], [fare, 'class'])
# df_titanic.pivot_table(index='sex', columns='class',
#                        aggfunc={'survived': sum, 'fare': 'mean'})
df_titanic.pivot_table('survived', index='sex', columns='class', margins=True)

# ageByGenderClassSurvival
grp = df_titanic.pivot_table('age', ['sex', 'survived'], 'class')
ax = grp.plot.bar(color=tableau20, legend=False, align='center')
#plt.legend(loc=2)
ax.set_axis_bgcolor('white')
plt.xticks(np.arange(0, 5), ['Female victims',
                             'Female survivors',
                             'Male victims',
                             'Male survivors'],
           rotation=0,
           ha='center',
           fontsize=14)
ax.set_xlabel("")
plt.yticks([])
plt.title("Average age of survivors and victims by gender and class\n",
          fontsize=22,
          loc='left')
rects = ax.patches
labels = [int(np.round(value, decimals=0)) for value in np.concatenate(grp.T.values)]
legends = ['First class', 'First class', 'First class', 'First class',
           'Second class', 'Second class', 'Second class', 'Second class',
           'Third class', 'Third class', 'Third class', 'Third class']
for rect, label, legend in zip(rects, labels, legends):
    height = rect.get_height()
    if float(label) >= 19:
        ax.text(rect.get_x() + rect.get_width()/2, height-2., label, ha='center',
                va='bottom', color='white', fontsize=16)
        ax.text(rect.get_x() + rect.get_width()/2, height-3, legend,
                ha='center', va='top', color='white', fontsize=16, rotation=90)
    else:
        ax.text(rect.get_x() + rect.get_width()/2, height-2., label, ha='center',
                va='bottom', color='white', fontsize=16)
        ax.text(rect.get_x() + rect.get_width()/2, height+3, legend,
                ha='center', va='bottom', color='black', fontsize=16, rotation=90)
ax.axis('tight')
plt.savefig("ageByGenderClassSurvival.png",
            bbox_inches='tight',
            dpi=300,
            format='png')
plt.savefig("ageByGenderClassSurvival.pdf",
            bbox_inches='tight',
            dpi=300,
            format='pdf')
