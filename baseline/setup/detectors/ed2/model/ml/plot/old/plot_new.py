import matplotlib.pyplot as plt
import numpy as np

dataset_name = 'Flights HoloClean'
labels10 = [4, 8, 12, 16, 26, 36, 46, 56, 66, 76, 86, 96, 106, 116, 126, 136, 146, 156, 166, 176, 186, 196, 206, 216, 226, 236, 246, 256, 266, 276, 286, 296, 306, 316, 326, 336, 346, 356, 366, 376, 386, 396, 406, 416]
fscore10 = [0.0, 0.0, 0.0, 0.0, 0.28217054263565894, 0.55498193082085689, 0.6541161455009572, 0.75082943428328364, 0.76079366431065321, 0.75427589170605935, 0.78415593705293285, 0.7903671535728386, 0.79057385131137836, 0.80683463654792942, 0.82335506816834614, 0.80785521755872158, 0.80393843800783871, 0.81216048794434392, 0.81197469071678152, 0.82749403341288785, 0.83174786201595086, 0.83083219645293316, 0.83663852346357737, 0.84520186650436191, 0.84531899786996645, 0.8450261780104712, 0.84677503932878861, 0.85232945091514145, 0.85283097418817666, 0.86246270959777804, 0.86546577457539897, 0.86590038314176243, 0.86601780907020087, 0.86557445914723796, 0.87559908314232138, 0.88402489626556013, 0.88402489626556013, 0.89476948521811039, 0.89983545865898806, 0.90705556126116871, 0.90817687493587762, 0.91181639615424381, 0.91209927611168562, 0.9193083573487032]
baseline = 0.763
baseline_reporter = 'HoloClean'
number_rows = 2376
number_cols = 6

labels5 = [4, 8, 12, 16, 21, 26, 31, 36, 41, 46, 51, 56, 61, 66, 71, 76, 81, 86, 91, 96, 101, 106, 111, 116, 121, 126, 131, 136, 141, 146, 151, 156, 161, 166, 171, 176, 181, 186, 191, 196, 201, 206, 211, 216, 221, 226, 231, 236, 241, 246, 251, 256, 261, 266, 271, 276, 281, 286, 291, 296, 301, 306, 311, 316, 321, 326, 331, 336, 341, 346, 351, 356, 361, 366, 371, 376, 381, 386, 391, 396, 401, 406, 411, 416]
fscore5 = [0.0, 0.0, 0.0, 0.0, 0.24972587719298245, 0.51054590570719605, 0.59246347941567068, 0.68219633943427616, 0.74899036968002475, 0.69810040705563092, 0.74233009708737852, 0.72493100275988964, 0.73395055448163604, 0.7803479258268019, 0.76980108499095845, 0.75064115210100602, 0.75113389863932167, 0.74812870613395555, 0.75792244608686854, 0.76964489077544262, 0.76964489077544262, 0.7807366595245383, 0.78251970115648339, 0.78238659583163062, 0.78362931210451103, 0.78633885402736348, 0.79126263141778086, 0.81436420722134994, 0.81436420722134994, 0.81831581016016508, 0.81917342139497651, 0.82207496785678957, 0.82207496785678957, 0.82120520202521596, 0.83750866422418058, 0.83886863112249921, 0.83886863112249921, 0.84593339262772993, 0.84736428009441389, 0.85667183863460039, 0.85667183863460039, 0.85588580367618305, 0.85292391942108348, 0.862464464268209, 0.86257596549696136, 0.8663279265194449, 0.87773137807128276, 0.87858379834519928, 0.87858379834519928, 0.87979788164415507, 0.88533488417175521, 0.8947062245491566, 0.8947062245491566, 0.90025375756392723, 0.90051742653519473, 0.89861886570672933, 0.89861886570672933, 0.90857030015797779, 0.90912682734097205, 0.90927218344965111, 0.90959832552576503, 0.90534151493813497, 0.90534151493813497, 0.9119496855345911, 0.9119496855345911, 0.91954485953076215, 0.91957725213890285, 0.921389028686462, 0.921389028686462, 0.9225937183383992, 0.92281199351701793, 0.92430844057148653, 0.92430844057148653, 0.92473769168684428, 0.9249243188698284, 0.92544212218649524, 0.92544212218649524, 0.92604177176874169, 0.92604177176874169, 0.92666733729001105, 0.92666733729001105, 0.92779168753129693, 0.92779168753129693, 0.93206823458160892]


labels15 = [4, 8, 12, 16, 31, 46, 61, 76, 91, 106, 121, 136, 151, 166, 181, 196, 211, 226, 241, 256, 271, 286, 301, 316, 331, 346, 361, 376, 391, 406, 421, 436, 451, 466, 481, 496, 511, 526, 541, 556]
fscore15 = [0.0, 0.0, 0.0, 0.0, 0.30286305622628495, 0.58783356519539642, 0.71859742565468265, 0.80721196130167105, 0.80717173492705219, 0.80619461104645951, 0.80721917931220255, 0.81461988304093569, 0.80360567702339858, 0.80566605219753562, 0.80292825768667642, 0.78254545454545466, 0.78433822763719663, 0.79368777801313273, 0.80390683696468812, 0.82745471877979027, 0.83663631494376001, 0.84646421745593547, 0.86164182238085252, 0.87502544270303273, 0.87815340618935767, 0.88778467908902703, 0.89558974358974353, 0.90382269215128108, 0.90382269215128108, 0.90709939148073038, 0.91180048661800495, 0.91732243229432808, 0.91732243229432808, 0.92268357537328705, 0.92279374169137962, 0.93060643060643056, 0.93060643060643056, 0.93654178089146101, 0.93654178089146101, 0.94052195023265217]

labels2 = [4, 8, 12, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126, 128, 130, 132, 134, 136, 138, 140, 142, 144, 146, 148, 150, 152, 154, 156, 158, 160, 162, 164, 166, 168, 170, 172, 174, 176, 178, 180, 182, 184, 186, 188, 190, 192, 194, 196, 198, 200, 202, 204, 206, 208, 210, 212, 214, 216, 218, 220, 222, 224, 226, 228, 230, 232, 234, 236, 238, 240, 242, 244, 246, 248, 250, 252, 254, 256, 258, 260, 262, 264, 266, 268, 270, 272, 274, 276, 278, 280, 282, 284, 286, 288, 290, 292, 294, 296, 298, 300, 302, 304, 306, 308, 310, 312, 314, 316, 318, 320, 322, 324, 326, 328, 330, 332, 334, 336, 338, 340, 342, 344, 346, 348, 350, 352, 354, 356, 358, 360, 362, 364, 366, 368, 370, 372, 374, 376, 378, 380, 382, 384, 386, 388, 390, 392, 394, 396, 398, 400, 402, 404, 406, 408, 410, 412, 414, 416, 418, 420, 422, 424, 426, 428, 430, 432, 434, 436, 438, 440, 442, 444, 446, 448, 450, 452, 454, 456, 458, 460, 462, 464, 466, 468, 470, 472, 474, 476, 478, 480, 482, 484, 486, 488]
fscore2 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.30153508771929821, 0.50682382133995052, 0.55680923642088698, 0.55680923642088698, 0.500411522633745, 0.47160493827160499, 0.42739520958083826, 0.42739520958083826, 0.44878829446730673, 0.45016993750685236, 0.47252747252747251, 0.47252747252747251, 0.54696198033874222, 0.56947763741891422, 0.6448650464777691, 0.65467865467865471, 0.66793381037567079, 0.67140635414352368, 0.67549816319715028, 0.70804647160068857, 0.70970464135021083, 0.70960884353741494, 0.7219707337614486, 0.76443829113924056, 0.76662346725152031, 0.76402457447879946, 0.76364002415945231, 0.76868815518761635, 0.76737040340965401, 0.77103948270794553, 0.77224234119604585, 0.79786727601637641, 0.79375886524822703, 0.79404513559643464, 0.7944217816146476, 0.7902910815586347, 0.7930646672914714, 0.7970701474316837, 0.7970701474316837, 0.8117842630217954, 0.81473993751148688, 0.81395348837209303, 0.81395348837209303, 0.8102678571428571, 0.80928844539774314, 0.81525091112980097, 0.81525091112980097, 0.81765799256505567, 0.81949592290585616, 0.81818181818181823, 0.81818181818181823, 0.82121043864519716, 0.82139215868013715, 0.8268124823229942, 0.82691039291435031, 0.83136814523675839, 0.83655994043186888, 0.83756721676246981, 0.83780024112028195, 0.84356508875739633, 0.84355121231676966, 0.84109740379303999, 0.84109740379303999, 0.83844165435745943, 0.83744570742075597, 0.83560485007989471, 0.83560485007989471, 0.83424168389400488, 0.83446072904922952, 0.83333333333333337, 0.83333333333333337, 0.8352584737739065, 0.83684857656294331, 0.83845126835781048, 0.83845126835781048, 0.84272770624701965, 0.84367859862909367, 0.84649039376070001, 0.84660009510223488, 0.84834620150605267, 0.85064935064935054, 0.85266277846830474, 0.85266277846830474, 0.85081920091980456, 0.85608153157443578, 0.85831426392067134, 0.85831426392067134, 0.85714285714285721, 0.85991995425957701, 0.85878191031647388, 0.85878191031647388, 0.86169400991987788, 0.86506047043138745, 0.87203881647797543, 0.87203881647797543, 0.87366412213740452, 0.87472569411315715, 0.87227233052080311, 0.87227233052080311, 0.87144240077444335, 0.87266389077176332, 0.86738491674828588, 0.86738491674828588, 0.86899262899262897, 0.87017957020900805, 0.87816270922537953, 0.87816270922537953, 0.87803451301550162, 0.87803451301550162, 0.87808687164470478, 0.87808687164470478, 0.87774846086191749, 0.87694566813509556, 0.87859020543277189, 0.87859020543277189, 0.87608125182233454, 0.87578784058954717, 0.88101847420312063, 0.88101847420312063, 0.8857033639143731, 0.88663735423437184, 0.88222903102259675, 0.88222903102259675, 0.8827320827320827, 0.88321940887648009, 0.88427527873894651, 0.88427527873894651, 0.88385542168674691, 0.88359992291385625, 0.8851734409532046, 0.8851734409532046, 0.88456375838926171, 0.88501007001054954, 0.88897352885164727, 0.88897352885164727, 0.8894414690130068, 0.88952654232424677, 0.89130225852955303, 0.89130225852955303, 0.89590211691590593, 0.896116504854369, 0.89567233384853162, 0.89567233384853162, 0.89583535579069995, 0.89562093407126908, 0.89691817215727954, 0.89691817215727954, 0.8984382578329615, 0.8984382578329615, 0.90204878048780479, 0.90204878048780479, 0.90211094747177223, 0.90211094747177223, 0.90292306188288207, 0.90292306188288207, 0.90477117818889974, 0.90477117818889974, 0.90450204638472032, 0.90450204638472032, 0.90504971729381944, 0.90504971729381944, 0.90725257631732459, 0.90725257631732459, 0.90824079485680886, 0.90845344760420721, 0.910051874327102, 0.910051874327102, 0.91382370341155594, 0.91382370341155594, 0.91282456313555138, 0.91282456313555138, 0.91291527422842544, 0.91291527422842544, 0.91586396697464123, 0.91586396697464123, 0.91484759095378565, 0.91484759095378565, 0.91581230283911663, 0.91581230283911663, 0.91535863302205456, 0.91535863302205456, 0.91915571485463965, 0.91915571485463965, 0.92413244506314007, 0.92413244506314007, 0.92529880478087645, 0.92529880478087645, 0.92273819055244199, 0.92273819055244199, 0.92355600200904064, 0.92355600200904064, 0.92887029288702927, 0.92887029288702927, 0.93124438678774579, 0.93124438678774579, 0.92742902838864449, 0.92753623188405798, 0.92809801164892536, 0.92809801164892536, 0.92894524959742353, 0.92894524959742353, 0.92878635907723173, 0.92878635907723173, 0.93052925346357018, 0.93052925346357018, 0.93209506690633104, 0.93209506690633104, 0.93361496680748346, 0.93361496680748346, 0.93476515455640308, 0.93476515455640308, 0.93433016771064858, 0.93433016771064858, 0.93272543059777091, 0.93272543059777091, 0.93433319821645722, 0.93433319821645722, 0.93487288564772619, 0.93487288564772619, 0.93450398627510345, 0.93450398627510345, 0.93522715825526348, 0.93522715825526348, 0.93713025167953468, 0.93713025167953468, 0.93645686708224418, 0.93645686708224418, 0.93422249166414062, 0.93422249166414062, 0.93507018075330706, 0.93507018075330706, 0.93744971842316971, 0.93744971842316971, 0.93904646952323478]

number_cells = number_rows * number_cols

fig = plt.figure()
ax = plt.subplot(111)
#ax2 = ax.twiny()


#ax.set_title(dataset_name)

ax.plot(labels15, fscore15, 'green', label="step: 15")
ax.plot(labels10, fscore10, 'blue', label="step: 10")
ax.plot(labels5, fscore5, 'red', label="step: 5")
ax.plot(labels2, fscore2, 'orange', label="step: 2")

ax.set_ylabel('percentage')
ax.set_xlabel('how much of the dataset was labeled')
#ax.plot((0, last), (baseline, baseline), 'k-', color='red')
#ax.text(last / 2, baseline - 0.05, r'baseline reported by ' + baseline_reporter, color='red')

ax.legend(loc=4)

#ax2.plot(rowsl, fscore)  # Create a dummy plot
#ax2.cla()


plt.show()