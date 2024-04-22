import matplotlib.pyplot as plt

blur_l = 7.379484563255651
blur_s = 9.987559782764844

l = [0.  , 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1 , 0.11, 0.12, 0.13, 
 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2 , 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27,
 0.28, 0.29, 0.3 , 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.4 , 0.41,
 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5 , 0.51, 0.52, 0.53, 0.54, 0.55,
 0.56, 0.57, 0.58, 0.59, 0.6 , 0.61, 0.62, 0.63, 0.64, 0.65, 0.66, 0.67, 0.68, 0.69,
 0.7 , 0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79, 0.8 , 0.81, 0.82, 0.83,
 0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 0.9 , 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97,
 0.98, 0.99]

snr_l = [0.11221509510442301, 4.493143061984278, 7.618683702017769, 8.93920623268288, 9.422228609067858, 9.608412337609721, 9.646971673432782, 9.609854280242102, 9.534652877343085, 9.443704305805618, 9.345660471912232, 9.247759079876122, 9.146729753234, 9.043171765157124, 8.945574236378837, 8.845101357052034, 8.757021782310044, 8.665220806055931, 8.578046231441295, 8.496280315827672, 8.412619170788759, 8.329136648327722, 8.235566852952534, 8.15509406372566, 8.076663972267868, 8.00190854551527, 7.929724588761258, 7.845219393526277, 7.760409807466817, 7.68154985912545, 7.589535060254082, 7.516377652428003, 7.4402410419013485, 7.364024215041528, 7.290930469143299, 7.219023242613522, 7.14890271748192, 7.089928582382308, 7.03265300418478, 6.972726683917387, 6.90912791173452, 6.844578077734507, 6.779096947773139, 6.711228013945627, 6.648420533542378, 6.589144148201127, 6.521857022440899, 6.4580681297216715, 6.3974568449731715, 6.341779458561049, 6.26695580133437, 6.206194603220054, 6.1380702538361405, 6.089052723793371, 6.001149088766286, 5.933633197799352, 5.869025418852977, 5.809493986032798, 5.7336995046392705, 5.669629298743067, 5.5953253182677365, 5.519301452632357, 5.456912328207948, 5.392961268193483, 5.304640867515234, 5.24375671873838, 5.159754203189167, 5.074461576035976, 5.0099571307943425, 4.937751944120147, 4.84102089134215, 4.774531798068599, 4.691419651191355, 4.6123801465355, 4.524136697665021, 4.4433343443292275, 4.359990573479252, 4.270215113001244, 4.193384330762402, 4.100710179127417, 4.024650075331781, 3.942297320622849, 3.885257258344838, 3.809526163931935, 3.718575070931239, 3.6324737939424794, 3.5618415293030803, 3.4447749283438664, 3.3428817865955587, 3.2069939525785616, 3.096397665276054, 2.9726376013962756, 2.7523551721552812, 2.587836989892163, 2.4048957970496603, 2.251871212645854, 2.0941065327861565, 1.7666763760885538, 1.478696181343038, 1.0567478134986532]

snr_s = [0.13694456009279263, 4.631631441149754, 8.401986737401169, 10.274788908548768, 11.241499953589468, 11.76012681193965, 12.035707977571375, 12.165198293422236, 12.2016259467574, 12.1979600699519, 12.169174803629872, 12.11075136752026, 12.040219335465776, 11.951919126065313, 11.882542188733392, 11.793383041529374, 11.711846922341898, 11.616455556020508, 11.513582346045634, 11.420825923862237, 11.338573238471753, 11.243746514826856, 11.144884527498597, 11.05818590888049, 10.963197879247815, 10.876890959835809, 10.771279385197833, 10.681186706368246, 10.589843282475957, 10.50158782717534, 10.41455493823256, 10.329058141365671, 10.239890742612786, 10.152081160847295, 10.058422454439139, 9.964339789553588, 9.882359913611795, 9.796867382177703, 9.710424716743793, 9.62268533144391, 9.539614185649326, 9.448806960313131, 9.360400172631202, 9.271635472752688, 9.184909584568777, 9.102836087869347, 9.013465604010731, 8.936100064701439, 8.84229616143619, 8.74693214822476, 8.643887082519932, 8.564149116835278, 8.468635599664424, 8.371683892632312, 8.264965993365841, 8.159835238491434, 8.053284793187688, 7.946956150043219, 7.847017479892396, 7.753458668562496, 7.635619689903068, 7.511218712888432, 7.402882005534391, 7.2648987338967475, 7.15591716425441, 7.027975003577147, 6.898409296919117, 6.781788085658363, 6.6426616271203285, 6.5196936373211125, 6.394088599114882, 6.278988183963236, 6.155072093962065, 6.027899593370148, 5.8734003866640085, 5.742912962478439, 5.605132373560186, 5.482629725110753, 5.334930935432086, 5.204837131259481, 5.059252936839915, 4.915764865693832, 4.768006023159237, 4.618420941695212, 4.4668746980089775, 4.309109759349543, 4.149081527280178, 3.9984859341999517, 3.8421250270740446, 3.6615669031934814, 3.489487499003408, 3.2903528883476496, 3.1183158488243565, 2.914867123377019, 2.7562981756339426, 2.5440096387113993, 2.286846005413489, 2.068155322505207, 1.7177369751185219, 1.3048878140813054]

fig, ax = plt.subplots()

plt.grid(axis='x', color='0.95')
plt.grid(axis='y', color='0.95')

plt.xscale('log')

plt.hlines(blur_l, l[0], l[-1], color='C0', linestyle='dashed')
plt.hlines(blur_s, l[0], l[-1], color='C1', linestyle='dashed')

plt.plot(l, snr_l, color='C0')
plt.plot(l, snr_s, color='C1')

plt.legend(('Lena blurred', 'Sabrina blurred', 'Lena de-blurred', 'Sabrina de-blurred'))

plt.xlabel('Limit')
plt.ylabel('SNR [dB]')

plt.show()
