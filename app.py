from flask import Flask, render_template, request, send_file
import numpy as np
import skfuzzy as sk
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

app = Flask(__name__)

image_directory = "static/Image"

bmi = ctrl.Antecedent(np.arange(0, 36, 1), 'bmi')
bmi['low'] = sk.trimf(bmi.universe, [1, 9.25, 18.5])
bmi['normal'] = sk.trimf(bmi.universe, [17.5, 21.7, 24.9])
bmi['high'] = sk.trimf(bmi.universe, [24.8, 27.5, 29.9])
bmi['very_high'] = sk.trimf(bmi.universe, [29.8, 32.5, 34.9])
bmi['very_very_high'] = sk.trimf(bmi.universe, [34.8, 35.5, 36.0])

bp = ctrl.Antecedent(np.arange(0, 110, 1), 'bp')
bp['low'] = sk.trimf(bp.universe, [0, 29, 59])
bp['normal'] = sk.trimf(bp.universe, [58, 70, 80])
bp['high'] = sk.trimf(bp.universe, [79, 85, 89])
bp['very_high'] = sk.trimf(bp.universe, [88, 95, 99])
bp['very_very_high'] = sk.trimf(bp.universe, [98, 100, 110])

sm = ctrl.Antecedent(np.arange(-1, 30, 1), 'sm')
sm['low'] = sk.trimf(sm.universe, [-1, 3.5, 6])
sm['moderate'] = sk.trimf(sm.universe, [5, 9, 12])
sm['high'] = sk.trimf(sm.universe, [11, 18, 23])
sm['very_high'] = sk.trimf(sm.universe, [22, 24, 30])

ex = ctrl.Antecedent(np.arange(0, 4.4, 0.1), 'ex')
ex['low'] = sk.trimf(ex.universe, [0, 0.7, 1.4])
ex['medium'] = sk.trimf(ex.universe, [1.3, 1.95, 2.4])
ex['normal'] = sk.trimf(ex.universe, [2.3, 2.9, 3.4])
ex['high'] = sk.trimf(ex.universe, [3.3, 3.5, 4.4])

stroke_risk = ctrl.Consequent(np.arange(1, 11, 1), 'stroke_risk')
stroke_risk.automf(3)

rule1 = ctrl.Rule(sm['moderate'] & bmi['high'] & ex['low'] & bp['normal'], stroke_risk['poor'])
rule2 = ctrl.Rule(sm['very_high'] & bmi['high'] & ex['low'] & bp['very_high'], stroke_risk['good'])
rule3 = ctrl.Rule(sm['low'] & bmi['low'] & ex['medium'] & bp['normal'], stroke_risk['poor'])
rule4 = ctrl.Rule(sm['low'] & bmi['low'] & ex['medium'] & bp['normal'], stroke_risk['poor'])
rule5 = ctrl.Rule(sm['moderate'] & bmi['high'] & ex['low'] & bp['normal'], stroke_risk['poor'])
rule6 = ctrl.Rule(sm['low'] & bmi['normal'] & ex['medium'] & bp['normal'], stroke_risk['poor'])
rule7 = ctrl.Rule(sm['low'] & bmi['low'] & ex['low'] & bp['low'], stroke_risk['poor'])
rule8 = ctrl.Rule(sm['low'] & bmi['low'] & ex['medium'] & bp['low'], stroke_risk['poor'])
rule9 = ctrl.Rule(sm['low'] & bmi['low'] & ex['normal'] & bp['low'], stroke_risk['poor'])
rule10 = ctrl.Rule(sm['low'] & bmi['low'] & ex['high'] & bp['low'], stroke_risk['poor'])
rule11 = ctrl.Rule(sm['low'] & bmi['normal'] & ex['high'] & bp['high'], stroke_risk['poor'])
rule12 = ctrl.Rule(sm['moderate'] & bmi['high'] & ex['low'] & bp['high'], stroke_risk['good'])
rule13 = ctrl.Rule(sm['low'] & bmi['high'] & ex['low'] & bp['high'], stroke_risk['average'])
rule14 = ctrl.Rule(sm['moderate'] & bmi['very_high'] & ex['low'] & bp['high'], stroke_risk['good'])
rule15 = ctrl.Rule(sm['very_high'] & bmi['very_high'] & ex['low'] & bp['high'], stroke_risk['good'])
rule16 = ctrl.Rule(sm['very_high'] & bmi['very_high'] & ex['low'] & bp['very_high'], stroke_risk['good'])
rule17 = ctrl.Rule(sm['very_high'] & bmi['very_high'] & ex['medium'] & bp['very_high'], stroke_risk['good'])
rule18 = ctrl.Rule(sm['low'] & bmi['normal'] & ex['high'] & bp['normal'], stroke_risk['poor'])
rule19 = ctrl.Rule(sm['low'] & bmi['normal'] & ex['medium'] & bp['high'], stroke_risk['poor'])
rule20 = ctrl.Rule(sm['low'] & bmi['normal'] & ex['normal'] & bp['high'], stroke_risk['poor'])
rule21 = ctrl.Rule(sm['low'] & bmi['normal'] & ex['normal'] & bp['normal'], stroke_risk['poor'])
rule22 = ctrl.Rule(sm['moderate'] & bmi['normal'] & ex['low'] & bp['high'], stroke_risk['good'])
rule23 = ctrl.Rule(sm['moderate'] & bmi['high'] & ex['normal'] & bp['high'], stroke_risk['poor'])
rule24 = ctrl.Rule(sm['moderate'] & bmi['high'] & ex['normal'] & bp['normal'], stroke_risk['poor'])
rule25 = ctrl.Rule(sm['low'] & bmi['high'] & ex['normal'] & bp['very_high'], stroke_risk['poor'])
rule26 = ctrl.Rule(sm['low'] & bmi['normal'] & ex['normal'] & bp['very_high'], stroke_risk['average'])
rule27 = ctrl.Rule(sm['moderate'] & bmi['normal'] & ex['normal'] & bp['very_high'], stroke_risk['average'])
rule28 = ctrl.Rule(sm['high'] & bmi['normal'] & ex['low'] & bp['high'], stroke_risk['good'])
rule29 = ctrl.Rule(sm['high'] & bmi['low'] & ex['low'] & bp['high'], stroke_risk['good'])
rule30 = ctrl.Rule(sm['high'] & bmi['high'] & ex['low'] & bp['high'], stroke_risk['good'])
rule31 = ctrl.Rule(sm['high'] & bmi['high'] & ex['normal'] & bp['high'], stroke_risk['good'])
rule32 = ctrl.Rule(sm['high'] & bmi['high'] & ex['normal'] & bp['very_high'], stroke_risk['good'])

rule33 = ctrl.Rule(sm['low'] & bmi['low'] & ex['low'] & bp['normal'], stroke_risk['poor'])
rule34 = ctrl.Rule(sm['low'] & bmi['high'] & ex['low'] & bp['low'], stroke_risk['average'])
rule35 = ctrl.Rule(sm['low'] & bmi['low'] & ex['low'] & bp['high'], stroke_risk['poor'])
rule36 = ctrl.Rule(sm['low'] & bmi['very_high'] & ex['low'] & bp['low'], stroke_risk['average'])
rule37 = ctrl.Rule(sm['low'] & bmi['low'] & ex['low'] & bp['very_high'], stroke_risk['good'])
rule38 = ctrl.Rule(sm['low'] & bmi['very_very_high'] & ex['low'] & bp['low'], stroke_risk['good'])
rule39 = ctrl.Rule(sm['low'] & bmi['low'] & ex['low'] & bp['very_very_high'], stroke_risk['good'])
rule40 = ctrl.Rule(sm['low'] & bmi['normal'] & ex['medium'] & bp['low'], stroke_risk['poor'])
rule41 = ctrl.Rule(sm['low'] & bmi['high'] & ex['medium'] & bp['low'], stroke_risk['poor'])
rule42 = ctrl.Rule(sm['low'] & bmi['high'] & ex['normal'] & bp['low'], stroke_risk['poor'])
rule43 = ctrl.Rule(sm['low'] & bmi['high'] & ex['high'] & bp['low'], stroke_risk['average'])
rule44 = ctrl.Rule(sm['low'] & bmi['high'] & ex['low'] & bp['normal'], stroke_risk['poor'])
rule45 = ctrl.Rule(sm['low'] & bmi['high'] & ex['low'] & bp['very_high'], stroke_risk['average'])
rule46 = ctrl.Rule(sm['low'] & bmi['high'] & ex['low'] & bp['very_very_high'], stroke_risk['good'])
rule47 = ctrl.Rule(sm['low'] & bmi['high'] & ex['medium'] & bp['normal'], stroke_risk['poor'])
rule48 = ctrl.Rule(sm['low'] & bmi['high'] & ex['medium'] & bp['high'], stroke_risk['poor'])
rule49 = ctrl.Rule(sm['low'] & bmi['high'] & ex['medium'] & bp['very_high'], stroke_risk['good'])
rule50 = ctrl.Rule(sm['low'] & bmi['high'] & ex['medium'] & bp['very_very_high'], stroke_risk['good'])
rule51 = ctrl.Rule(sm['low'] & bmi['high'] & ex['normal'] & bp['normal'], stroke_risk['poor'])
rule52 = ctrl.Rule(sm['low'] & bmi['high'] & ex['normal'] & bp['high'], stroke_risk['average'])
rule53 = ctrl.Rule(sm['low'] & bmi['high'] & ex['normal'] & bp['very_very_high'], stroke_risk['good'])
rule54 = ctrl.Rule(sm['low'] & bmi['very_high'] & ex['medium'] & bp['low'], stroke_risk['average'])
rule55 = ctrl.Rule(sm['low'] & bmi['very_high'] & ex['normal'] & bp['low'], stroke_risk['average'])
rule56 = ctrl.Rule(sm['low'] & bmi['very_high'] & ex['high'] & bp['low'], stroke_risk['average'])
rule57 = ctrl.Rule(sm['low'] & bmi['very_high'] & ex['low'] & bp['normal'], stroke_risk['good'])
rule58 = ctrl.Rule(sm['low'] & bmi['very_high'] & ex['low'] & bp['high'], stroke_risk['good'])
rule59 = ctrl.Rule(sm['low'] & bmi['very_high'] & ex['low'] & bp['very_high'], stroke_risk['good'])
rule60 = ctrl.Rule(sm['low'] & bmi['very_high'] & ex['low'] & bp['very_very_high'], stroke_risk['good'])
rule61 = ctrl.Rule(sm['low'] & bmi['very_high'] & ex['medium'] & bp['high'], stroke_risk['good'])
rule62 = ctrl.Rule(sm['low'] & bmi['very_high'] & ex['medium'] & bp['very_high'], stroke_risk['good'])
rule63 = ctrl.Rule(sm['low'] & bmi['very_high'] & ex['medium'] & bp['very_very_high'], stroke_risk['good'])
rule64 = ctrl.Rule(sm['low'] & bmi['very_high'] & ex['normal'] & bp['high'], stroke_risk['good'])
rule65 = ctrl.Rule(sm['low'] & bmi['very_high'] & ex['normal'] & bp['very_high'], stroke_risk['good'])
rule66 = ctrl.Rule(sm['low'] & bmi['very_high'] & ex['normal'] & bp['very_very_high'], stroke_risk['good'])
rule67 = ctrl.Rule(sm['low'] & bmi['very_high'] & ex['high'] & bp['high'], stroke_risk['average'])
rule68 = ctrl.Rule(sm['low'] & bmi['very_high'] & ex['high'] & bp['very_high'], stroke_risk['good'])
rule69 = ctrl.Rule(sm['low'] & bmi['very_high'] & ex['high'] & bp['very_very_high'], stroke_risk['good'])

rule70 = ctrl.Rule(sm['low'] & bmi['very_very_high'] & ex['medium'] & bp['low'], stroke_risk['good'])
rule71 = ctrl.Rule(sm['low'] & bmi['very_very_high'] & ex['normal'] & bp['low'], stroke_risk['good'])
rule72 = ctrl.Rule(sm['low'] & bmi['very_very_high'] & ex['high'] & bp['low'], stroke_risk['good'])
rule73 = ctrl.Rule(sm['low'] & bmi['very_very_high'] & ex['low'] & bp['normal'], stroke_risk['good'])
rule74 = ctrl.Rule(sm['low'] & bmi['very_very_high'] & ex['low'] & bp['high'], stroke_risk['good'])
rule75 = ctrl.Rule(sm['low'] & bmi['very_very_high'] & ex['low'] & bp['very_high'], stroke_risk['good'])
rule76 = ctrl.Rule(sm['low'] & bmi['very_very_high'] & ex['low'] & bp['very_very_high'], stroke_risk['good'])
rule77 = ctrl.Rule(sm['low'] & bmi['very_very_high'] & ex['medium'] & bp['high'], stroke_risk['good'])
rule78 = ctrl.Rule(sm['low'] & bmi['very_very_high'] & ex['medium'] & bp['very_high'], stroke_risk['good'])
rule79 = ctrl.Rule(sm['low'] & bmi['very_very_high'] & ex['medium'] & bp['very_very_high'], stroke_risk['good'])
rule80 = ctrl.Rule(sm['low'] & bmi['very_very_high'] & ex['normal'] & bp['high'], stroke_risk['good'])
rule81 = ctrl.Rule(sm['low'] & bmi['very_very_high'] & ex['normal'] & bp['very_high'], stroke_risk['good'])
rule82 = ctrl.Rule(sm['low'] & bmi['very_very_high'] & ex['normal'] & bp['very_very_high'], stroke_risk['good'])
rule83 = ctrl.Rule(sm['low'] & bmi['very_very_high'] & ex['high'] & bp['high'], stroke_risk['good'])
rule84 = ctrl.Rule(sm['low'] & bmi['very_very_high'] & ex['high'] & bp['very_high'], stroke_risk['good'])
rule85 = ctrl.Rule(sm['low'] & bmi['very_very_high'] & ex['high'] & bp['very_very_high'], stroke_risk['good'])

rule86 = ctrl.Rule(sm['low'] & bmi['low'] & ex['medium'] & bp['high'], stroke_risk['poor'])
rule87 = ctrl.Rule(sm['low'] & bmi['low'] & ex['medium'] & bp['very_high'], stroke_risk['average'])
rule88 = ctrl.Rule(sm['low'] & bmi['low'] & ex['medium'] & bp['very_very_high'], stroke_risk['good'])
rule89 = ctrl.Rule(sm['low'] & bmi['low'] & ex['normal'] & bp['normal'], stroke_risk['poor'])
rule90 = ctrl.Rule(sm['low'] & bmi['low'] & ex['normal'] & bp['high'], stroke_risk['poor'])
rule91 = ctrl.Rule(sm['low'] & bmi['low'] & ex['normal'] & bp['very_high'], stroke_risk['average'])
rule92 = ctrl.Rule(sm['low'] & bmi['low'] & ex['normal'] & bp['very_very_high'], stroke_risk['good'])
rule93 = ctrl.Rule(sm['low'] & bmi['low'] & ex['high'] & bp['normal'], stroke_risk['poor'])
rule94 = ctrl.Rule(sm['low'] & bmi['low'] & ex['high'] & bp['high'], stroke_risk['poor'])
rule95 = ctrl.Rule(sm['low'] & bmi['low'] & ex['high'] & bp['very_high'], stroke_risk['average'])
rule96 = ctrl.Rule(sm['low'] & bmi['low'] & ex['high'] & bp['very_very_high'], stroke_risk['good'])

rule97 = ctrl.Rule(sm['low'] & bmi['normal'] & ex['high'] & bp['very_high'], stroke_risk['average'])
rule98 = ctrl.Rule(sm['low'] & bmi['normal'] & ex['high'] & bp['very_very_high'], stroke_risk['good'])
rule99 = ctrl.Rule(sm['low'] & bmi['normal'] & ex['high'] & bp['low'], stroke_risk['poor'])
rule100 = ctrl.Rule(sm['low'] & bmi['normal'] & ex['medium'] & bp['very_high'], stroke_risk['good'])
rule101 = ctrl.Rule(sm['low'] & bmi['normal'] & ex['medium'] & bp['very_very_high'], stroke_risk['good'])
rule102 = ctrl.Rule(sm['low'] & bmi['normal'] & ex['normal'] & bp['low'], stroke_risk['poor'])
rule103 = ctrl.Rule(sm['low'] & bmi['normal'] & ex['normal'] & bp['very_very_high'], stroke_risk['good'])
rule104 = ctrl.Rule(sm['low'] & bmi['very_high'] & ex['normal'] & bp['normal'], stroke_risk['average'])
rule105 = ctrl.Rule(sm['low'] & bmi['very_very_high'] & ex['normal'] & bp['normal'], stroke_risk['good'])
rule106 = ctrl.Rule(sm['low'] & bmi['very_very_high'] & ex['medium'] & bp['normal'], stroke_risk['good'])
rule107 = ctrl.Rule(sm['low'] & bmi['very_very_high'] & ex['high'] & bp['normal'], stroke_risk['good'])
rule108 = ctrl.Rule(sm['low'] & bmi['high'] & ex['high'] & bp['normal'], stroke_risk['average'])
rule109 = ctrl.Rule(sm['low'] & bmi['high'] & ex['high'] & bp['high'], stroke_risk['average'])
rule110 = ctrl.Rule(sm['low'] & bmi['high'] & ex['high'] & bp['very_high'], stroke_risk['good'])
rule111 = ctrl.Rule(sm['low'] & bmi['high'] & ex['high'] & bp['very_very_high'], stroke_risk['good'])

rule112 = ctrl.Rule(sm['moderate'] & bmi['low'] & ex['low'] & bp['low'], stroke_risk['average'])
rule113 = ctrl.Rule(sm['moderate'] & bmi['normal'] & ex['low'] & bp['low'], stroke_risk['average'])
rule114 = ctrl.Rule(sm['moderate'] & bmi['low'] & ex['medium'] & bp['low'], stroke_risk['average'])
rule115 = ctrl.Rule(sm['moderate'] & bmi['low'] & ex['low'] & bp['normal'], stroke_risk['average'])
rule116 = ctrl.Rule(sm['moderate'] & bmi['high'] & ex['low'] & bp['low'], stroke_risk['average'])
rule117 = ctrl.Rule(sm['moderate'] & bmi['low'] & ex['normal'] & bp['low'], stroke_risk['average'])
rule118 = ctrl.Rule(sm['moderate'] & bmi['low'] & ex['low'] & bp['high'], stroke_risk['average'])
rule119 = ctrl.Rule(sm['moderate'] & bmi['very_high'] & ex['low'] & bp['low'], stroke_risk['average'])
rule120 = ctrl.Rule(sm['moderate'] & bmi['low'] & ex['high'] & bp['low'], stroke_risk['average'])
rule121 = ctrl.Rule(sm['moderate'] & bmi['low'] & ex['low'] & bp['very_high'], stroke_risk['good'])
rule122 = ctrl.Rule(sm['moderate'] & bmi['very_very_high'] & ex['low'] & bp['low'], stroke_risk['good'])
rule123 = ctrl.Rule(sm['moderate'] & bmi['low'] & ex['low'] & bp['very_very_high'], stroke_risk['good'])
rule124 = ctrl.Rule(sm['moderate'] & bmi['normal'] & ex['medium'] & bp['low'], stroke_risk['average'])
rule125 = ctrl.Rule(sm['moderate'] & bmi['low'] & ex['medium'] & bp['normal'], stroke_risk['average'])
rule126 = ctrl.Rule(sm['moderate'] & bmi['normal'] & ex['medium'] & bp['normal'], stroke_risk['average'])
rule127 = ctrl.Rule(sm['moderate'] & bmi['high'] & ex['medium'] & bp['low'], stroke_risk['average'])
rule128 = ctrl.Rule(sm['moderate'] & bmi['high'] & ex['normal'] & bp['low'], stroke_risk['average'])
rule129 = ctrl.Rule(sm['moderate'] & bmi['high'] & ex['high'] & bp['low'], stroke_risk['average'])
rule130 = ctrl.Rule(sm['moderate'] & bmi['high'] & ex['low'] & bp['normal'], stroke_risk['average'])
rule131 = ctrl.Rule(sm['moderate'] & bmi['high'] & ex['low'] & bp['high'], stroke_risk['average'])
rule132 = ctrl.Rule(sm['moderate'] & bmi['high'] & ex['low'] & bp['very_high'], stroke_risk['average'])
rule133 = ctrl.Rule(sm['moderate'] & bmi['high'] & ex['low'] & bp['very_very_high'], stroke_risk['good'])
rule134 = ctrl.Rule(sm['moderate'] & bmi['high'] & ex['medium'] & bp['normal'], stroke_risk['average'])
rule135 = ctrl.Rule(sm['moderate'] & bmi['high'] & ex['medium'] & bp['high'], stroke_risk['average'])
rule136 = ctrl.Rule(sm['moderate'] & bmi['high'] & ex['medium'] & bp['very_high'], stroke_risk['good'])
rule137 = ctrl.Rule(sm['moderate'] & bmi['high'] & ex['medium'] & bp['very_very_high'], stroke_risk['good'])
rule138 = ctrl.Rule(sm['moderate'] & bmi['high'] & ex['normal'] & bp['normal'], stroke_risk['average'])
rule139 = ctrl.Rule(sm['moderate'] & bmi['high'] & ex['normal'] & bp['high'], stroke_risk['average'])
rule140 = ctrl.Rule(sm['moderate'] & bmi['high'] & ex['normal'] & bp['very_high'], stroke_risk['good'])
rule141 = ctrl.Rule(sm['moderate'] & bmi['high'] & ex['normal'] & bp['very_very_high'], stroke_risk['good'])
rule142 = ctrl.Rule(sm['moderate'] & bmi['very_high'] & ex['medium'] & bp['low'], stroke_risk['average'])
rule143 = ctrl.Rule(sm['moderate'] & bmi['very_high'] & ex['normal'] & bp['low'], stroke_risk['average'])
rule144 = ctrl.Rule(sm['moderate'] & bmi['very_high'] & ex['high'] & bp['low'], stroke_risk['average'])
rule145 = ctrl.Rule(sm['moderate'] & bmi['very_high'] & ex['low'] & bp['normal'], stroke_risk['good'])
rule146 = ctrl.Rule(sm['moderate'] & bmi['very_high'] & ex['low'] & bp['high'], stroke_risk['good'])
rule147 = ctrl.Rule(sm['moderate'] & bmi['very_high'] & ex['low'] & bp['very_high'], stroke_risk['good'])
rule148 = ctrl.Rule(sm['moderate'] & bmi['very_high'] & ex['low'] & bp['very_very_high'], stroke_risk['good'])
rule149 = ctrl.Rule(sm['moderate'] & bmi['very_high'] & ex['medium'] & bp['high'], stroke_risk['good'])
rule150 = ctrl.Rule(sm['moderate'] & bmi['very_high'] & ex['medium'] & bp['very_high'], stroke_risk['good'])
rule151 = ctrl.Rule(sm['moderate'] & bmi['very_high'] & ex['medium'] & bp['very_very_high'], stroke_risk['good'])
rule152 = ctrl.Rule(sm['moderate'] & bmi['very_high'] & ex['normal'] & bp['high'], stroke_risk['good'])
rule153 = ctrl.Rule(sm['moderate'] & bmi['very_high'] & ex['normal'] & bp['very_high'], stroke_risk['good'])
rule154 = ctrl.Rule(sm['moderate'] & bmi['very_high'] & ex['normal'] & bp['very_very_high'], stroke_risk['good'])
rule155 = ctrl.Rule(sm['moderate'] & bmi['very_high'] & ex['high'] & bp['high'], stroke_risk['average'])
rule156 = ctrl.Rule(sm['moderate'] & bmi['very_high'] & ex['high'] & bp['very_high'], stroke_risk['good'])
rule157 = ctrl.Rule(sm['moderate'] & bmi['very_high'] & ex['high'] & bp['very_very_high'], stroke_risk['good'])

rule158 = ctrl.Rule(sm['moderate'] & bmi['very_very_high'] & ex['medium'] & bp['low'], stroke_risk['good'])
rule159 = ctrl.Rule(sm['moderate'] & bmi['very_very_high'] & ex['normal'] & bp['low'], stroke_risk['good'])
rule160 = ctrl.Rule(sm['moderate'] & bmi['very_very_high'] & ex['high'] & bp['low'], stroke_risk['good'])
rule161 = ctrl.Rule(sm['moderate'] & bmi['very_very_high'] & ex['low'] & bp['normal'], stroke_risk['good'])
rule162 = ctrl.Rule(sm['moderate'] & bmi['very_very_high'] & ex['low'] & bp['high'], stroke_risk['good'])
rule163 = ctrl.Rule(sm['moderate'] & bmi['very_very_high'] & ex['low'] & bp['very_high'], stroke_risk['good'])
rule164 = ctrl.Rule(sm['moderate'] & bmi['very_very_high'] & ex['low'] & bp['very_very_high'], stroke_risk['good'])
rule165 = ctrl.Rule(sm['moderate'] & bmi['very_very_high'] & ex['medium'] & bp['high'], stroke_risk['good'])
rule166 = ctrl.Rule(sm['moderate'] & bmi['very_very_high'] & ex['medium'] & bp['very_high'], stroke_risk['good'])
rule167 = ctrl.Rule(sm['moderate'] & bmi['very_very_high'] & ex['medium'] & bp['very_very_high'], stroke_risk['good'])
rule168 = ctrl.Rule(sm['moderate'] & bmi['very_very_high'] & ex['normal'] & bp['high'], stroke_risk['good'])
rule169 = ctrl.Rule(sm['moderate'] & bmi['very_very_high'] & ex['normal'] & bp['very_high'], stroke_risk['good'])
rule170 = ctrl.Rule(sm['moderate'] & bmi['very_very_high'] & ex['normal'] & bp['very_very_high'], stroke_risk['good'])
rule171 = ctrl.Rule(sm['moderate'] & bmi['very_very_high'] & ex['high'] & bp['high'], stroke_risk['good'])
rule172 = ctrl.Rule(sm['moderate'] & bmi['very_very_high'] & ex['high'] & bp['very_high'], stroke_risk['good'])
rule173 = ctrl.Rule(sm['moderate'] & bmi['very_very_high'] & ex['high'] & bp['very_very_high'], stroke_risk['good'])

rule174 = ctrl.Rule(sm['moderate'] & bmi['low'] & ex['medium'] & bp['high'], stroke_risk['average'])
rule175 = ctrl.Rule(sm['moderate'] & bmi['low'] & ex['medium'] & bp['very_high'], stroke_risk['average'])
rule176 = ctrl.Rule(sm['moderate'] & bmi['low'] & ex['medium'] & bp['very_very_high'], stroke_risk['good'])
rule401 = ctrl.Rule(sm['moderate'] & bmi['low'] & ex['normal'] & bp['normal'], stroke_risk['average'])
rule177 = ctrl.Rule(sm['moderate'] & bmi['low'] & ex['normal'] & bp['high'], stroke_risk['average'])
rule178 = ctrl.Rule(sm['moderate'] & bmi['low'] & ex['normal'] & bp['very_high'], stroke_risk['average'])
rule179 = ctrl.Rule(sm['moderate'] & bmi['low'] & ex['normal'] & bp['very_very_high'], stroke_risk['good'])
rule180 = ctrl.Rule(sm['moderate'] & bmi['low'] & ex['high'] & bp['normal'], stroke_risk['average'])
rule181 = ctrl.Rule(sm['moderate'] & bmi['low'] & ex['high'] & bp['high'], stroke_risk['average'])
rule182 = ctrl.Rule(sm['moderate'] & bmi['low'] & ex['high'] & bp['very_high'], stroke_risk['average'])
rule183 = ctrl.Rule(sm['moderate'] & bmi['low'] & ex['high'] & bp['very_very_high'], stroke_risk['good'])

rule184 = ctrl.Rule(sm['moderate'] & bmi['normal'] & ex['high'] & bp['very_high'], stroke_risk['average'])
rule185 = ctrl.Rule(sm['moderate'] & bmi['normal'] & ex['high'] & bp['very_very_high'], stroke_risk['good'])
rule186 = ctrl.Rule(sm['moderate'] & bmi['normal'] & ex['high'] & bp['low'], stroke_risk['average'])
rule187 = ctrl.Rule(sm['moderate'] & bmi['normal'] & ex['medium'] & bp['very_high'], stroke_risk['good'])
rule188 = ctrl.Rule(sm['moderate'] & bmi['normal'] & ex['medium'] & bp['very_very_high'], stroke_risk['good'])
rule189 = ctrl.Rule(sm['moderate'] & bmi['normal'] & ex['normal'] & bp['low'], stroke_risk['average'])
rule190 = ctrl.Rule(sm['moderate'] & bmi['normal'] & ex['normal'] & bp['very_very_high'], stroke_risk['good'])
rule191 = ctrl.Rule(sm['moderate'] & bmi['very_high'] & ex['normal'] & bp['normal'], stroke_risk['average'])
rule192 = ctrl.Rule(sm['moderate'] & bmi['very_very_high'] & ex['normal'] & bp['normal'], stroke_risk['good'])
rule193 = ctrl.Rule(sm['moderate'] & bmi['very_very_high'] & ex['medium'] & bp['normal'], stroke_risk['good'])
rule194 = ctrl.Rule(sm['moderate'] & bmi['very_very_high'] & ex['high'] & bp['normal'], stroke_risk['good'])
rule195 = ctrl.Rule(sm['moderate'] & bmi['high'] & ex['high'] & bp['normal'], stroke_risk['average'])
rule196 = ctrl.Rule(sm['moderate'] & bmi['high'] & ex['high'] & bp['high'], stroke_risk['average'])
rule197 = ctrl.Rule(sm['moderate'] & bmi['high'] & ex['high'] & bp['very_high'], stroke_risk['good'])
rule198 = ctrl.Rule(sm['moderate'] & bmi['high'] & ex['high'] & bp['very_very_high'], stroke_risk['good'])

rule199 = ctrl.Rule(sm['moderate'] & bmi['normal'] & ex['high'] & bp['high'], stroke_risk['average'])
rule200 = ctrl.Rule(sm['moderate'] & bmi['normal'] & ex['high'] & bp['normal'], stroke_risk['average'])
rule201 = ctrl.Rule(sm['moderate'] & bmi['normal'] & ex['medium'] & bp['high'], stroke_risk['average'])
rule202 = ctrl.Rule(sm['moderate'] & bmi['normal'] & ex['normal'] & bp['high'], stroke_risk['average'])
rule203 = ctrl.Rule(sm['moderate'] & bmi['normal'] & ex['normal'] & bp['normal'], stroke_risk['average'])
rule204 = ctrl.Rule(sm['moderate'] & bmi['normal'] & ex['normal'] & bp['very_high'], stroke_risk['average'])
rule205 = ctrl.Rule(sm['moderate'] & bmi['very_high'] & ex['medium'] & bp['normal'], stroke_risk['average'])

rule206 = ctrl.Rule(sm['high'] & bmi['low'] & ex['low'] & bp['low'], stroke_risk['good'])
rule207 = ctrl.Rule(sm['high'] & bmi['normal'] & ex['low'] & bp['low'], stroke_risk['good'])
rule208 = ctrl.Rule(sm['high'] & bmi['low'] & ex['medium'] & bp['low'], stroke_risk['good'])
rule209 = ctrl.Rule(sm['high'] & bmi['low'] & ex['low'] & bp['normal'], stroke_risk['good'])
rule210 = ctrl.Rule(sm['high'] & bmi['high'] & ex['low'] & bp['low'], stroke_risk['good'])
rule211 = ctrl.Rule(sm['high'] & bmi['low'] & ex['normal'] & bp['low'], stroke_risk['good'])
rule212 = ctrl.Rule(sm['high'] & bmi['low'] & ex['low'] & bp['high'], stroke_risk['good'])
rule213 = ctrl.Rule(sm['high'] & bmi['very_high'] & ex['low'] & bp['low'], stroke_risk['good'])
rule214 = ctrl.Rule(sm['high'] & bmi['low'] & ex['high'] & bp['low'], stroke_risk['good'])
rule215 = ctrl.Rule(sm['high'] & bmi['low'] & ex['low'] & bp['very_high'], stroke_risk['good'])
rule216 = ctrl.Rule(sm['high'] & bmi['very_very_high'] & ex['low'] & bp['low'], stroke_risk['good'])
rule217 = ctrl.Rule(sm['high'] & bmi['low'] & ex['low'] & bp['very_very_high'], stroke_risk['good'])
rule218 = ctrl.Rule(sm['high'] & bmi['normal'] & ex['medium'] & bp['low'], stroke_risk['good'])
rule219 = ctrl.Rule(sm['high'] & bmi['low'] & ex['medium'] & bp['normal'], stroke_risk['good'])
rule220 = ctrl.Rule(sm['high'] & bmi['normal'] & ex['medium'] & bp['normal'], stroke_risk['good'])
rule221 = ctrl.Rule(sm['high'] & bmi['high'] & ex['medium'] & bp['low'], stroke_risk['good'])
rule222 = ctrl.Rule(sm['high'] & bmi['high'] & ex['normal'] & bp['low'], stroke_risk['good'])
rule223 = ctrl.Rule(sm['high'] & bmi['high'] & ex['high'] & bp['low'], stroke_risk['good'])
rule224 = ctrl.Rule(sm['high'] & bmi['high'] & ex['low'] & bp['normal'], stroke_risk['good'])
rule225 = ctrl.Rule(sm['high'] & bmi['high'] & ex['low'] & bp['high'], stroke_risk['good'])
rule226 = ctrl.Rule(sm['high'] & bmi['high'] & ex['low'] & bp['very_high'], stroke_risk['good'])
rule227 = ctrl.Rule(sm['high'] & bmi['high'] & ex['low'] & bp['very_very_high'], stroke_risk['good'])
rule228 = ctrl.Rule(sm['high'] & bmi['high'] & ex['medium'] & bp['normal'], stroke_risk['good'])
rule229 = ctrl.Rule(sm['high'] & bmi['high'] & ex['medium'] & bp['high'], stroke_risk['good'])
rule230 = ctrl.Rule(sm['high'] & bmi['high'] & ex['medium'] & bp['very_high'], stroke_risk['good'])
rule231 = ctrl.Rule(sm['high'] & bmi['high'] & ex['medium'] & bp['very_very_high'], stroke_risk['good'])
rule232 = ctrl.Rule(sm['high'] & bmi['high'] & ex['normal'] & bp['normal'], stroke_risk['good'])
rule233 = ctrl.Rule(sm['high'] & bmi['high'] & ex['normal'] & bp['high'], stroke_risk['good'])
rule234 = ctrl.Rule(sm['high'] & bmi['high'] & ex['normal'] & bp['very_high'], stroke_risk['good'])
rule235 = ctrl.Rule(sm['high'] & bmi['high'] & ex['normal'] & bp['very_very_high'], stroke_risk['good'])
rule236 = ctrl.Rule(sm['high'] & bmi['very_high'] & ex['medium'] & bp['low'], stroke_risk['good'])
rule237 = ctrl.Rule(sm['high'] & bmi['very_high'] & ex['normal'] & bp['low'], stroke_risk['good'])
rule238 = ctrl.Rule(sm['high'] & bmi['very_high'] & ex['high'] & bp['low'], stroke_risk['good'])
rule239 = ctrl.Rule(sm['high'] & bmi['very_high'] & ex['low'] & bp['normal'], stroke_risk['good'])
rule240 = ctrl.Rule(sm['high'] & bmi['very_high'] & ex['low'] & bp['high'], stroke_risk['good'])
rule241 = ctrl.Rule(sm['high'] & bmi['very_high'] & ex['low'] & bp['very_high'], stroke_risk['good'])
rule242 = ctrl.Rule(sm['high'] & bmi['very_high'] & ex['low'] & bp['very_very_high'], stroke_risk['good'])
rule243 = ctrl.Rule(sm['high'] & bmi['very_high'] & ex['medium'] & bp['high'], stroke_risk['good'])
rule244 = ctrl.Rule(sm['high'] & bmi['very_high'] & ex['medium'] & bp['very_high'], stroke_risk['good'])
rule245 = ctrl.Rule(sm['high'] & bmi['very_high'] & ex['medium'] & bp['very_very_high'], stroke_risk['good'])
rule246 = ctrl.Rule(sm['high'] & bmi['very_high'] & ex['normal'] & bp['high'], stroke_risk['good'])
rule247 = ctrl.Rule(sm['high'] & bmi['very_high'] & ex['normal'] & bp['very_high'], stroke_risk['good'])
rule248 = ctrl.Rule(sm['high'] & bmi['very_high'] & ex['normal'] & bp['very_very_high'], stroke_risk['good'])
rule249 = ctrl.Rule(sm['high'] & bmi['very_high'] & ex['high'] & bp['high'], stroke_risk['good'])
rule250 = ctrl.Rule(sm['high'] & bmi['very_high'] & ex['high'] & bp['very_high'], stroke_risk['good'])
rule251 = ctrl.Rule(sm['high'] & bmi['very_high'] & ex['high'] & bp['very_very_high'], stroke_risk['good'])

rule252 = ctrl.Rule(sm['high'] & bmi['very_very_high'] & ex['medium'] & bp['low'], stroke_risk['good'])
rule253 = ctrl.Rule(sm['high'] & bmi['very_very_high'] & ex['normal'] & bp['low'], stroke_risk['good'])
rule254 = ctrl.Rule(sm['high'] & bmi['very_very_high'] & ex['high'] & bp['low'], stroke_risk['good'])
rule255 = ctrl.Rule(sm['high'] & bmi['very_very_high'] & ex['low'] & bp['normal'], stroke_risk['good'])
rule256 = ctrl.Rule(sm['high'] & bmi['very_very_high'] & ex['low'] & bp['high'], stroke_risk['good'])
rule257 = ctrl.Rule(sm['high'] & bmi['very_very_high'] & ex['low'] & bp['very_high'], stroke_risk['good'])
rule258 = ctrl.Rule(sm['high'] & bmi['very_very_high'] & ex['low'] & bp['very_very_high'], stroke_risk['good'])
rule259 = ctrl.Rule(sm['high'] & bmi['very_very_high'] & ex['medium'] & bp['high'], stroke_risk['good'])
rule260 = ctrl.Rule(sm['high'] & bmi['very_very_high'] & ex['medium'] & bp['very_high'], stroke_risk['good'])
rule261 = ctrl.Rule(sm['high'] & bmi['very_very_high'] & ex['medium'] & bp['very_very_high'], stroke_risk['good'])
rule262 = ctrl.Rule(sm['high'] & bmi['very_very_high'] & ex['normal'] & bp['high'], stroke_risk['good'])
rule263 = ctrl.Rule(sm['high'] & bmi['very_very_high'] & ex['normal'] & bp['very_high'], stroke_risk['good'])
rule264 = ctrl.Rule(sm['high'] & bmi['very_very_high'] & ex['normal'] & bp['very_very_high'], stroke_risk['good'])
rule265 = ctrl.Rule(sm['high'] & bmi['very_very_high'] & ex['high'] & bp['high'], stroke_risk['good'])
rule266 = ctrl.Rule(sm['high'] & bmi['very_very_high'] & ex['high'] & bp['very_high'], stroke_risk['good'])
rule267 = ctrl.Rule(sm['high'] & bmi['very_very_high'] & ex['high'] & bp['very_very_high'], stroke_risk['good'])

rule268 = ctrl.Rule(sm['high'] & bmi['low'] & ex['medium'] & bp['high'], stroke_risk['good'])
rule269 = ctrl.Rule(sm['high'] & bmi['low'] & ex['medium'] & bp['very_high'], stroke_risk['good'])
rule270 = ctrl.Rule(sm['high'] & bmi['low'] & ex['medium'] & bp['very_very_high'], stroke_risk['good'])
rule271 = ctrl.Rule(sm['high'] & bmi['low'] & ex['normal'] & bp['normal'], stroke_risk['good'])
rule272 = ctrl.Rule(sm['high'] & bmi['low'] & ex['normal'] & bp['high'], stroke_risk['good'])
rule273 = ctrl.Rule(sm['high'] & bmi['low'] & ex['normal'] & bp['very_high'], stroke_risk['good'])
rule274 = ctrl.Rule(sm['high'] & bmi['low'] & ex['normal'] & bp['very_very_high'], stroke_risk['good'])
rule275 = ctrl.Rule(sm['high'] & bmi['low'] & ex['high'] & bp['normal'], stroke_risk['good'])
rule276 = ctrl.Rule(sm['high'] & bmi['low'] & ex['high'] & bp['high'], stroke_risk['good'])
rule277 = ctrl.Rule(sm['high'] & bmi['low'] & ex['high'] & bp['very_high'], stroke_risk['good'])
rule278 = ctrl.Rule(sm['high'] & bmi['low'] & ex['high'] & bp['very_very_high'], stroke_risk['good'])

rule279 = ctrl.Rule(sm['high'] & bmi['normal'] & ex['high'] & bp['very_high'], stroke_risk['good'])
rule280 = ctrl.Rule(sm['high'] & bmi['normal'] & ex['high'] & bp['very_very_high'], stroke_risk['good'])
rule281 = ctrl.Rule(sm['high'] & bmi['normal'] & ex['high'] & bp['low'], stroke_risk['good'])
rule282 = ctrl.Rule(sm['high'] & bmi['normal'] & ex['medium'] & bp['very_high'], stroke_risk['good'])
rule283 = ctrl.Rule(sm['high'] & bmi['normal'] & ex['medium'] & bp['very_very_high'], stroke_risk['good'])
rule284 = ctrl.Rule(sm['high'] & bmi['normal'] & ex['normal'] & bp['low'], stroke_risk['good'])
rule285 = ctrl.Rule(sm['high'] & bmi['normal'] & ex['normal'] & bp['very_very_high'], stroke_risk['good'])
rule286 = ctrl.Rule(sm['high'] & bmi['very_high'] & ex['normal'] & bp['normal'], stroke_risk['good'])
rule287 = ctrl.Rule(sm['high'] & bmi['very_very_high'] & ex['normal'] & bp['normal'], stroke_risk['good'])
rule288 = ctrl.Rule(sm['high'] & bmi['very_very_high'] & ex['medium'] & bp['normal'], stroke_risk['good'])
rule289 = ctrl.Rule(sm['high'] & bmi['very_very_high'] & ex['high'] & bp['normal'], stroke_risk['good'])
rule290 = ctrl.Rule(sm['high'] & bmi['high'] & ex['high'] & bp['normal'], stroke_risk['good'])
rule291 = ctrl.Rule(sm['high'] & bmi['high'] & ex['high'] & bp['high'], stroke_risk['good'])
rule292 = ctrl.Rule(sm['high'] & bmi['high'] & ex['high'] & bp['very_high'], stroke_risk['good'])
rule293 = ctrl.Rule(sm['high'] & bmi['high'] & ex['high'] & bp['very_very_high'], stroke_risk['good'])

rule294 = ctrl.Rule(sm['high'] & bmi['normal'] & ex['high'] & bp['high'], stroke_risk['good'])
rule295 = ctrl.Rule(sm['high'] & bmi['normal'] & ex['high'] & bp['normal'], stroke_risk['good'])
rule296 = ctrl.Rule(sm['high'] & bmi['normal'] & ex['medium'] & bp['high'], stroke_risk['good'])
rule297 = ctrl.Rule(sm['high'] & bmi['normal'] & ex['normal'] & bp['high'], stroke_risk['good'])
rule298 = ctrl.Rule(sm['high'] & bmi['normal'] & ex['normal'] & bp['normal'], stroke_risk['good'])
rule299 = ctrl.Rule(sm['high'] & bmi['normal'] & ex['normal'] & bp['very_high'], stroke_risk['good'])
rule300 = ctrl.Rule(sm['moderate'] & bmi['very_high'] & ex['medium'] & bp['normal'], stroke_risk['good'])

rule301 = ctrl.Rule(sm['very_high'] & bmi['low'] & ex['low'] & bp['low'], stroke_risk['good'])
rule302 = ctrl.Rule(sm['very_high'] & bmi['normal'] & ex['low'] & bp['low'], stroke_risk['good'])
rule303 = ctrl.Rule(sm['very_high'] & bmi['low'] & ex['medium'] & bp['low'], stroke_risk['good'])
rule304 = ctrl.Rule(sm['very_high'] & bmi['low'] & ex['low'] & bp['normal'], stroke_risk['good'])
rule305 = ctrl.Rule(sm['very_high'] & bmi['high'] & ex['low'] & bp['low'], stroke_risk['good'])
rule306 = ctrl.Rule(sm['very_high'] & bmi['low'] & ex['normal'] & bp['low'], stroke_risk['good'])
rule307 = ctrl.Rule(sm['very_high'] & bmi['low'] & ex['low'] & bp['high'], stroke_risk['good'])
rule308 = ctrl.Rule(sm['very_high'] & bmi['very_high'] & ex['low'] & bp['low'], stroke_risk['good'])
rule309 = ctrl.Rule(sm['very_high'] & bmi['low'] & ex['high'] & bp['low'], stroke_risk['good'])
rule310 = ctrl.Rule(sm['very_high'] & bmi['low'] & ex['low'] & bp['very_high'], stroke_risk['good'])
rule311 = ctrl.Rule(sm['very_high'] & bmi['very_very_high'] & ex['low'] & bp['low'], stroke_risk['good'])
rule312 = ctrl.Rule(sm['very_high'] & bmi['low'] & ex['low'] & bp['very_very_high'], stroke_risk['good'])
rule313 = ctrl.Rule(sm['very_high'] & bmi['normal'] & ex['medium'] & bp['low'], stroke_risk['good'])
rule314 = ctrl.Rule(sm['very_high'] & bmi['low'] & ex['medium'] & bp['normal'], stroke_risk['good'])
rule315 = ctrl.Rule(sm['very_high'] & bmi['normal'] & ex['medium'] & bp['normal'], stroke_risk['good'])
rule316 = ctrl.Rule(sm['very_high'] & bmi['high'] & ex['medium'] & bp['low'], stroke_risk['good'])
rule317 = ctrl.Rule(sm['very_high'] & bmi['high'] & ex['normal'] & bp['low'], stroke_risk['good'])
rule318 = ctrl.Rule(sm['very_high'] & bmi['high'] & ex['high'] & bp['low'], stroke_risk['good'])
rule319 = ctrl.Rule(sm['very_high'] & bmi['high'] & ex['low'] & bp['normal'], stroke_risk['good'])
rule320 = ctrl.Rule(sm['very_high'] & bmi['high'] & ex['low'] & bp['high'], stroke_risk['good'])
rule321 = ctrl.Rule(sm['very_high'] & bmi['high'] & ex['low'] & bp['very_high'], stroke_risk['good'])
rule322 = ctrl.Rule(sm['very_high'] & bmi['high'] & ex['low'] & bp['very_very_high'], stroke_risk['good'])
rule323 = ctrl.Rule(sm['very_high'] & bmi['high'] & ex['medium'] & bp['normal'], stroke_risk['good'])
rule324 = ctrl.Rule(sm['very_high'] & bmi['high'] & ex['medium'] & bp['high'], stroke_risk['good'])
rule325 = ctrl.Rule(sm['very_high'] & bmi['high'] & ex['medium'] & bp['very_high'], stroke_risk['good'])
rule326 = ctrl.Rule(sm['very_high'] & bmi['high'] & ex['medium'] & bp['very_very_high'], stroke_risk['good'])
rule327 = ctrl.Rule(sm['very_high'] & bmi['high'] & ex['normal'] & bp['normal'], stroke_risk['good'])
rule328 = ctrl.Rule(sm['very_high'] & bmi['high'] & ex['normal'] & bp['high'], stroke_risk['good'])
rule329 = ctrl.Rule(sm['very_high'] & bmi['high'] & ex['normal'] & bp['very_high'], stroke_risk['good'])
rule330 = ctrl.Rule(sm['very_high'] & bmi['high'] & ex['normal'] & bp['very_very_high'], stroke_risk['good'])
rule331 = ctrl.Rule(sm['very_high'] & bmi['very_high'] & ex['medium'] & bp['low'], stroke_risk['good'])
rule332 = ctrl.Rule(sm['very_high'] & bmi['very_high'] & ex['normal'] & bp['low'], stroke_risk['good'])
rule333 = ctrl.Rule(sm['very_high'] & bmi['very_high'] & ex['high'] & bp['low'], stroke_risk['good'])
rule334 = ctrl.Rule(sm['very_high'] & bmi['very_high'] & ex['low'] & bp['normal'], stroke_risk['good'])
rule335 = ctrl.Rule(sm['very_high'] & bmi['very_high'] & ex['low'] & bp['high'], stroke_risk['good'])
rule336 = ctrl.Rule(sm['very_high'] & bmi['very_high'] & ex['low'] & bp['very_high'], stroke_risk['good'])
rule337 = ctrl.Rule(sm['very_high'] & bmi['very_high'] & ex['low'] & bp['very_very_high'], stroke_risk['good'])
rule338 = ctrl.Rule(sm['very_high'] & bmi['very_high'] & ex['medium'] & bp['high'], stroke_risk['good'])
rule339 = ctrl.Rule(sm['very_high'] & bmi['very_high'] & ex['medium'] & bp['very_high'], stroke_risk['good'])
rule340 = ctrl.Rule(sm['very_high'] & bmi['very_high'] & ex['medium'] & bp['very_very_high'], stroke_risk['good'])
rule341 = ctrl.Rule(sm['very_high'] & bmi['very_high'] & ex['normal'] & bp['high'], stroke_risk['good'])
rule342 = ctrl.Rule(sm['very_high'] & bmi['very_high'] & ex['normal'] & bp['very_high'], stroke_risk['good'])
rule343 = ctrl.Rule(sm['very_high'] & bmi['very_high'] & ex['normal'] & bp['very_very_high'], stroke_risk['good'])
rule344 = ctrl.Rule(sm['very_high'] & bmi['very_high'] & ex['high'] & bp['high'], stroke_risk['good'])
rule345 = ctrl.Rule(sm['very_high'] & bmi['very_high'] & ex['high'] & bp['very_high'], stroke_risk['good'])
rule346 = ctrl.Rule(sm['very_high'] & bmi['very_high'] & ex['high'] & bp['very_very_high'], stroke_risk['good'])

rule347 = ctrl.Rule(sm['very_high'] & bmi['very_very_high'] & ex['medium'] & bp['low'], stroke_risk['good'])
rule348 = ctrl.Rule(sm['very_high'] & bmi['very_very_high'] & ex['normal'] & bp['low'], stroke_risk['good'])
rule349 = ctrl.Rule(sm['very_high'] & bmi['very_very_high'] & ex['high'] & bp['low'], stroke_risk['good'])
rule350 = ctrl.Rule(sm['very_high'] & bmi['very_very_high'] & ex['low'] & bp['normal'], stroke_risk['good'])
rule351 = ctrl.Rule(sm['very_high'] & bmi['very_very_high'] & ex['low'] & bp['high'], stroke_risk['good'])
rule352 = ctrl.Rule(sm['very_high'] & bmi['very_very_high'] & ex['low'] & bp['very_high'], stroke_risk['good'])
rule353 = ctrl.Rule(sm['very_high'] & bmi['very_very_high'] & ex['low'] & bp['very_very_high'], stroke_risk['good'])
rule354 = ctrl.Rule(sm['very_high'] & bmi['very_very_high'] & ex['medium'] & bp['high'], stroke_risk['good'])
rule355 = ctrl.Rule(sm['very_high'] & bmi['very_very_high'] & ex['medium'] & bp['very_high'], stroke_risk['good'])
rule356 = ctrl.Rule(sm['very_high'] & bmi['very_very_high'] & ex['medium'] & bp['very_very_high'], stroke_risk['good'])
rule357 = ctrl.Rule(sm['very_high'] & bmi['very_very_high'] & ex['normal'] & bp['high'], stroke_risk['good'])
rule358 = ctrl.Rule(sm['very_high'] & bmi['very_very_high'] & ex['normal'] & bp['very_high'], stroke_risk['good'])
rule359 = ctrl.Rule(sm['very_high'] & bmi['very_very_high'] & ex['normal'] & bp['very_very_high'], stroke_risk['good'])
rule360 = ctrl.Rule(sm['very_high'] & bmi['very_very_high'] & ex['high'] & bp['high'], stroke_risk['good'])
rule361 = ctrl.Rule(sm['very_high'] & bmi['very_very_high'] & ex['high'] & bp['very_high'], stroke_risk['good'])
rule362 = ctrl.Rule(sm['very_high'] & bmi['very_very_high'] & ex['high'] & bp['very_very_high'], stroke_risk['good'])

rule363 = ctrl.Rule(sm['very_high'] & bmi['low'] & ex['medium'] & bp['high'], stroke_risk['good'])
rule364 = ctrl.Rule(sm['very_high'] & bmi['low'] & ex['medium'] & bp['very_high'], stroke_risk['good'])
rule365 = ctrl.Rule(sm['very_high'] & bmi['low'] & ex['medium'] & bp['very_very_high'], stroke_risk['good'])
rule366 = ctrl.Rule(sm['very_high'] & bmi['low'] & ex['normal'] & bp['normal'], stroke_risk['good'])
rule367 = ctrl.Rule(sm['very_high'] & bmi['low'] & ex['normal'] & bp['high'], stroke_risk['good'])
rule368 = ctrl.Rule(sm['very_high'] & bmi['low'] & ex['normal'] & bp['very_high'], stroke_risk['good'])
rule369 = ctrl.Rule(sm['very_high'] & bmi['low'] & ex['normal'] & bp['very_very_high'], stroke_risk['good'])
rule370 = ctrl.Rule(sm['very_high'] & bmi['low'] & ex['high'] & bp['normal'], stroke_risk['good'])
rule371 = ctrl.Rule(sm['very_high'] & bmi['low'] & ex['high'] & bp['high'], stroke_risk['good'])
rule372 = ctrl.Rule(sm['very_high'] & bmi['low'] & ex['high'] & bp['very_high'], stroke_risk['good'])
rule373 = ctrl.Rule(sm['very_high'] & bmi['low'] & ex['high'] & bp['very_very_high'], stroke_risk['good'])

rule374 = ctrl.Rule(sm['very_high'] & bmi['normal'] & ex['high'] & bp['very_high'], stroke_risk['good'])
rule375 = ctrl.Rule(sm['very_high'] & bmi['normal'] & ex['high'] & bp['very_very_high'], stroke_risk['good'])
rule376 = ctrl.Rule(sm['very_high'] & bmi['normal'] & ex['high'] & bp['low'], stroke_risk['good'])
rule377 = ctrl.Rule(sm['very_high'] & bmi['normal'] & ex['medium'] & bp['very_high'], stroke_risk['good'])
rule378 = ctrl.Rule(sm['very_high'] & bmi['normal'] & ex['medium'] & bp['very_very_high'], stroke_risk['good'])
rule379 = ctrl.Rule(sm['very_high'] & bmi['normal'] & ex['normal'] & bp['low'], stroke_risk['good'])
rule380 = ctrl.Rule(sm['very_high'] & bmi['normal'] & ex['normal'] & bp['very_very_high'], stroke_risk['good'])
rule381 = ctrl.Rule(sm['very_high'] & bmi['very_high'] & ex['normal'] & bp['normal'], stroke_risk['good'])
rule382 = ctrl.Rule(sm['very_high'] & bmi['very_very_high'] & ex['normal'] & bp['normal'], stroke_risk['good'])
rule383 = ctrl.Rule(sm['very_high'] & bmi['very_very_high'] & ex['medium'] & bp['normal'], stroke_risk['good'])
rule384 = ctrl.Rule(sm['very_high'] & bmi['very_very_high'] & ex['high'] & bp['normal'], stroke_risk['good'])
rule385 = ctrl.Rule(sm['very_high'] & bmi['high'] & ex['high'] & bp['normal'], stroke_risk['good'])
rule386 = ctrl.Rule(sm['very_high'] & bmi['high'] & ex['high'] & bp['high'], stroke_risk['good'])
rule387 = ctrl.Rule(sm['very_high'] & bmi['high'] & ex['high'] & bp['very_high'], stroke_risk['good'])
rule388 = ctrl.Rule(sm['very_high'] & bmi['high'] & ex['high'] & bp['very_very_high'], stroke_risk['good'])

rule389 = ctrl.Rule(sm['very_high'] & bmi['normal'] & ex['high'] & bp['high'], stroke_risk['good'])
rule390 = ctrl.Rule(sm['very_high'] & bmi['normal'] & ex['high'] & bp['normal'], stroke_risk['good'])
rule391 = ctrl.Rule(sm['very_high'] & bmi['normal'] & ex['medium'] & bp['high'], stroke_risk['good'])
rule392 = ctrl.Rule(sm['very_high'] & bmi['normal'] & ex['normal'] & bp['high'], stroke_risk['good'])
rule393 = ctrl.Rule(sm['very_high'] & bmi['normal'] & ex['normal'] & bp['normal'], stroke_risk['good'])
rule394 = ctrl.Rule(sm['very_high'] & bmi['normal'] & ex['normal'] & bp['very_high'], stroke_risk['good'])
rule395 = ctrl.Rule(sm['moderate'] & bmi['very_high'] & ex['medium'] & bp['normal'], stroke_risk['good'])
rule396 = ctrl.Rule(sm['moderate'] & bmi['normal'] & ex['low'] & bp['normal'], stroke_risk['average'])
rule397 = ctrl.Rule(sm['moderate'] & bmi['normal'] & ex['low'] & bp['high'], stroke_risk['average'])
rule398 = ctrl.Rule(sm['moderate'] & bmi['normal'] & ex['low'] & bp['very_high'], stroke_risk['good'])
rule399 = ctrl.Rule(sm['moderate'] & bmi['normal'] & ex['low'] & bp['very_very_high'], stroke_risk['good'])
rule400 = ctrl.Rule(sm['moderate'] & bmi['normal'] & ex['low'] & bp['low'], stroke_risk['average'])
tipping = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10, rule11,
                              rule12, rule13, rule14, rule15, rule16, rule17, rule18, rule19, rule20, rule21, rule22,
                              rule23
                                 , rule24, rule25, rule26, rule27, rule28, rule29, rule30, rule31, rule32, rule33,
                              rule34, rule35, rule36, rule37, rule38
                                 , rule39, rule40, rule41, rule42, rule43, rule44, rule45, rule46, rule47, rule48,
                              rule49, rule50,
                              rule51, rule52, rule53, rule54, rule55, rule56, rule57, rule58, rule59, rule60,
                              rule61, rule62, rule63, rule64, rule65, rule66, rule67, rule68, rule69, rule70,
                              rule71, rule72, rule73, rule74, rule75, rule76, rule77, rule78, rule79,
                              rule80, rule81, rule82, rule83, rule84, rule85, rule86, rule87, rule88, rule89,
                              rule90, rule91, rule92, rule93, rule94, rule95, rule96, rule97, rule98, rule99, rule100,
                              rule101, rule102, rule103, rule104, rule105, rule106, rule107, rule108, rule109, rule110,
                              rule111,
                              rule112, rule113, rule114, rule115, rule116, rule117, rule118, rule119, rule120, rule121,
                              rule122,
                              rule123, rule124, rule125, rule126, rule127, rule128, rule129, rule130, rule131, rule132,
                              rule133,
                              rule134, rule135, rule136, rule137, rule138, rule139, rule140, rule141, rule142, rule143,
                              rule144,
                              rule145, rule146, rule147, rule148, rule149, rule150, rule151, rule152, rule153, rule154,
                              rule155,
                              rule156, rule157, rule158, rule159, rule160, rule161, rule162, rule163, rule164, rule165,
                              rule166,
                              rule167, rule168, rule169, rule170, rule171, rule172, rule173, rule174, rule175, rule176,
                              rule177, rule178,
                              rule179, rule180, rule181, rule182, rule183, rule184, rule185, rule186, rule187, rule188,
                              rule189, rule190,
                              rule191, rule192, rule193, rule194, rule195, rule196, rule197, rule198, rule199, rule200,
                              rule201, rule202, rule203, rule204, rule205,
                              rule207, rule208, rule209, rule210, rule211, rule212, rule213, rule214, rule215, rule216,
                              rule217, rule218, rule219, rule220, rule221, rule222, rule223, rule224, rule225, rule226,
                              rule227, rule228, rule229, rule230, rule231, rule232, rule233, rule234, rule235, rule236,
                              rule237, rule238, rule239, rule240, rule241, rule242, rule243, rule244, rule245, rule246,
                              rule247, rule248, rule249, rule250, rule251, rule252, rule253, rule254, rule255, rule256,
                              rule257, rule258, rule259, rule260, rule261, rule262, rule263, rule264, rule265, rule266,
                              rule267, rule268, rule269, rule270, rule271, rule272, rule273, rule274, rule275, rule276,
                              rule277, rule278, rule279, rule280, rule281, rule282, rule283, rule284, rule285, rule286,
                              rule287, rule288, rule289, rule290, rule291, rule292, rule293, rule294, rule295, rule296,
                              rule297, rule298, rule299, rule300, rule301, rule302, rule303, rule304, rule305, rule306,
                              rule307, rule308, rule309, rule310, rule311, rule312, rule313, rule314, rule315, rule316,
                              rule317, rule318, rule319, rule320, rule321, rule322, rule323, rule324, rule325, rule326,
                              rule327, rule328, rule329, rule330, rule331, rule332, rule333, rule334, rule335, rule336,
                              rule337, rule338, rule339, rule340, rule341, rule342, rule343, rule344, rule345, rule346,
                              rule347, rule348, rule349, rule350, rule351, rule352, rule353, rule354, rule355, rule356,
                              rule357, rule358, rule359, rule360, rule361, rule362, rule363, rule364, rule365, rule366,
                              rule367, rule368, rule369, rule370, rule371, rule372, rule373, rule374, rule375,
                              rule376, rule377, rule378, rule379, rule380, rule381, rule382, rule383, rule384,
                              rule385, rule386, rule387, rule388, rule389, rule390, rule391, rule392, rule393, rule394,
                              rule395,
                              rule396, rule397, rule398, rule399, rule400, rule401])
Tip = ctrl.ControlSystemSimulation(tipping)


@app.route('/')
def index():
    try:
        return render_template('index.html')
    except Exception as e:
        # Handle exceptions here (e.g., log the error, display an error page)
        return f"An error occurred: {str(e)}", 500  # Return a 500 Internal Server Error response


@app.route('/submit', methods=['POST'])
def submit():
    global sm_value, bmi_value, ex_value, bp_value
    try:
        sm_value = float(request.form.get('sm'))
        bmi_value = float(request.form.get('bmi'))
        ex_value = float(request.form.get('ex'))
        bp_value = float(request.form.get('bp'))

        Tip.input['sm'] = sm_value
        Tip.input['bmi'] = bmi_value
        Tip.input['ex'] = ex_value
        Tip.input['bp'] = bp_value
        Tip.compute()

        output = Tip.output['stroke_risk']
        if output < 4:
            result = "STROKE RISK IS LOW"
            return get_image('low')
        elif 4 <= output <= 6:
            result = "STROKE RISK IS MODERATE"
            return get_image('moderate')
        else:
            result = "STROKE RISK IS HIGH"
            return get_image('high')
    except Exception as e:
        # Handle exceptions here (e.g., log the error, display an error page)
        with open('form_data.txt', 'a') as file:
            file.write(f'Smoking: {sm_value}, BMI: {bmi_value}, Exercise:{ex_value}, BP:{bp_value} \n')
        return send_file("static/Image/confused-heart.png", mimetype='image/png')


def get_image(img_type):
    # Specify the path to your image file
    if img_type == 'low':
        image_path = f"{image_directory}/low_stroke.png"
    elif img_type == 'moderate':
        image_path = f"{image_directory}/moderate_stroke.png"
    elif img_type == 'high':
        image_path = f"{image_directory}/high_stroke.png"

    # Specify the MIME type for the image (e.g., 'image/png')
    mime_type = 'image/jpg'

    return send_file(image_path, mimetype=mime_type)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
