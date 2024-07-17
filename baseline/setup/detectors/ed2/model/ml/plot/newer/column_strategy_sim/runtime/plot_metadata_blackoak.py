import numpy as np

labels_all = [4, 8, 12, 16, 20, 24, 28, 38, 48, 58, 68, 78, 88, 98, 108, 118, 128, 138, 148, 158, 168, 178, 188, 198, 208, 218, 228, 238, 248, 258, 268, 278, 288, 298, 308, 318, 328, 338, 348, 358, 368, 378, 388, 398, 408, 418, 428, 438, 448, 458, 468, 478, 488, 498, 508, 518, 528, 538, 548, 558, 568]
runtime = []
runtime.append([91.73482298851013, 95.8991858959198, 100.09737610816956, 104.30228090286255, 108.46844410896301, 112.68790793418884, 116.87783312797546, 120.38111305236816, 123.83899211883545, 127.32474493980408, 130.85110998153687, 134.43580889701843, 138.05239391326904, 141.65350604057312, 145.32548809051514, 149.0402319431305, 152.73189902305603, 156.60587310791016, 160.20871996879578, 163.83281111717224, 167.41151404380798, 171.15283703804016, 174.98459005355835, 178.86919498443604, 182.78393507003784, 186.45506310462952, 190.2663290500641, 194.19672203063965, 198.15002989768982, 202.31979608535767, 206.03006505966187, 209.682030916214, 213.82919192314148, 217.4688069820404, 221.24353408813477, 225.10004806518555, 229.17420101165771, 233.45073699951172, 237.85391902923584, 242.04749608039856, 246.36183309555054, 250.7779860496521, 254.7591540813446, 258.4766240119934, 262.25803208351135, 265.9875349998474, 269.8416440486908, 274.0480740070343, 278.3425440788269, 282.8986439704895, 286.87784910202026, 291.03242802619934, 294.8141849040985, 298.94793605804443, 302.8147339820862, 307.4484689235687, 311.2898659706116, 315.46159291267395, 319.65852308273315, 324.2655439376831, 329.1112611293793])
runtime.append([88.72132611274719, 92.91800618171692, 97.18136715888977, 101.48582005500793, 105.65375113487244, 109.92832112312317, 114.17717909812927, 117.74931502342224, 121.32139706611633, 124.82873606681824, 128.3928201198578, 132.00104999542236, 135.61214303970337, 139.1743721961975, 142.8275430202484, 146.45766019821167, 150.17801117897034, 153.8921880722046, 157.61261510849, 161.43678903579712, 165.00298714637756, 168.66726517677307, 172.29139018058777, 176.09676599502563, 179.84352898597717, 183.66979503631592, 187.5410351753235, 191.24556016921997, 194.99335598945618, 198.98047518730164, 202.81538605690002, 206.99218106269836, 210.75254106521606, 214.66148209571838, 218.6533579826355, 222.62399911880493, 226.99207305908203, 231.04549908638, 235.28181505203247, 239.5515501499176, 243.3512761592865, 247.19244503974915, 251.04627513885498, 255.0299961566925, 259.1423251628876, 263.56701016426086, 267.4789021015167, 271.6033990383148, 275.75176215171814, 279.8631331920624, 284.4098381996155, 288.50631403923035, 293.3628661632538, 297.2424681186676, 301.16303300857544, 305.05255007743835, 309.36611318588257, 313.884211063385, 318.9364140033722, 323.38892912864685, 327.56269812583923])
runtime.append([87.48272514343262, 91.6254551410675, 95.76312804222107, 99.92873001098633, 104.06636905670166, 108.19075298309326, 112.3256139755249, 115.7471661567688, 119.2177791595459, 122.64762210845947, 126.13075017929077, 129.67046904563904, 133.21746706962585, 136.80076599121094, 140.4773030281067, 144.11180114746094, 147.75515699386597, 151.3597640991211, 155.0801591873169, 158.68203806877136, 162.3151731491089, 165.91042017936707, 169.6831510066986, 173.68074798583984, 177.67831015586853, 181.32568907737732, 185.05430102348328, 188.82386207580566, 192.46789813041687, 196.35354804992676, 200.47503805160522, 204.151349067688, 208.14112305641174, 211.8602740764618, 215.5432481765747, 219.83894205093384, 223.6126639842987, 227.3565001487732, 231.8037359714508, 235.6467890739441, 239.35915613174438, 243.4795160293579, 247.4007580280304, 251.42449712753296, 255.76556396484375, 260.1525230407715, 264.1445469856262, 268.32786297798157, 272.6476581096649, 277.21287417411804, 281.3538591861725, 285.1954550743103, 289.30440616607666, 294.0074429512024, 298.04050612449646, 301.8676221370697, 306.07747316360474, 310.9329171180725, 314.7486560344696, 319.8481550216675, 324.79185009002686])
runtime.append([88.77317094802856, 93.04064989089966, 97.24959683418274, 101.46376585960388, 105.67623591423035, 109.92498588562012, 114.18487095832825, 117.69928503036499, 121.17368292808533, 124.70221900939941, 128.20768404006958, 131.86074090003967, 135.46542191505432, 139.08632397651672, 142.8059949874878, 146.48044896125793, 150.13910388946533, 153.7845799922943, 157.37704181671143, 161.25863099098206, 164.85600304603577, 168.5159649848938, 172.23073387145996, 175.98661589622498, 179.8463418483734, 183.6604118347168, 187.57864093780518, 191.28555393218994, 195.26322484016418, 198.95959401130676, 202.9608769416809, 207.11508297920227, 210.82996702194214, 214.62001395225525, 218.93749690055847, 222.77513790130615, 226.67368483543396, 230.60989594459534, 234.65994381904602, 238.5352008342743, 242.26952195167542, 246.7541308403015, 250.92980098724365, 254.7206380367279, 258.7708308696747, 262.53143191337585, 266.8572618961334, 271.07338786125183, 275.16592502593994, 279.2073760032654, 283.8127989768982, 287.96012783050537, 292.5866138935089, 296.4034788608551, 300.75087785720825, 304.5772490501404, 309.00759100914, 313.4734878540039, 317.2801208496094, 321.37916803359985, 326.05491304397583])
runtime.append([88.8717908859253, 93.071448802948, 97.28941178321838, 101.46550178527832, 105.71057391166687, 109.97418999671936, 114.23781490325928, 117.78254294395447, 121.31048393249512, 124.84422898292542, 128.40114378929138, 131.9875807762146, 135.65311980247498, 139.22430396080017, 142.91074085235596, 146.54623985290527, 150.47538495063782, 154.1543369293213, 158.1507248878479, 162.21423077583313, 165.8376908302307, 169.4879059791565, 173.19079494476318, 176.99552083015442, 180.86561393737793, 184.6715259552002, 188.50534081459045, 192.4049038887024, 196.13166880607605, 200.13099694252014, 203.91349482536316, 207.69547986984253, 211.78930187225342, 215.9362759590149, 219.70422387123108, 224.03308176994324, 227.9404308795929, 232.21123099327087, 236.5422728061676, 240.66065382957458, 244.44766783714294, 248.24968194961548, 252.00781679153442, 256.00263690948486, 259.94796681404114, 264.3056149482727, 268.2919428348541, 272.86826491355896, 277.14855194091797, 281.63006591796875, 286.1774458885193, 290.9374408721924, 295.0585389137268, 299.8123879432678, 303.788813829422, 308.66576886177063, 312.5287389755249, 317.5042519569397, 322.1321008205414, 327.34271597862244, 331.410591840744])
runtime.append([88.35835695266724, 92.63263702392578, 96.91329216957092, 101.19092106819153, 105.46137595176697, 109.74559903144836, 114.09624195098877, 117.640958070755, 121.26158213615417, 124.85518097877502, 128.47209000587463, 132.16195797920227, 135.8481261730194, 139.52377104759216, 143.27155709266663, 147.04936599731445, 151.09116005897522, 154.84667897224426, 158.79029512405396, 162.5018391609192, 166.21836709976196, 169.8741271495819, 174.09355807304382, 177.8834011554718, 181.7647831439972, 185.74282217025757, 189.50233507156372, 193.33720302581787, 197.66362404823303, 201.5040991306305, 205.34273600578308, 209.43493700027466, 213.25352501869202, 217.12828516960144, 221.51324105262756, 225.81383514404297, 229.70843601226807, 233.6741189956665, 237.551922082901, 241.4456009864807, 245.362291097641, 249.40007400512695, 253.5040409564972, 257.6003370285034, 262.2036530971527, 266.43902015686035, 270.96077013015747, 275.54693508148193, 280.2425661087036, 285.0159389972687, 289.1813941001892, 293.3143301010132, 297.79167199134827, 302.51858711242676, 306.48451495170593, 311.15742111206055, 315.07427406311035, 320.0039939880371, 324.93389415740967, 328.78634214401245, 333.0306830406189])
runtime.append([89.17896103858948, 93.5076220035553, 97.82107996940613, 102.07196807861328, 106.29604387283325, 110.5314130783081, 114.79524493217468, 118.33881306648254, 121.8610348701477, 125.36224389076233, 128.9631998538971, 132.58796906471252, 136.2230670452118, 139.90442490577698, 143.69292306900024, 147.64721703529358, 151.36198496818542, 155.15472507476807, 159.17196989059448, 162.90213894844055, 166.55116391181946, 170.19370007514954, 174.36403393745422, 178.09905099868774, 181.99064302444458, 185.81353187561035, 189.65952491760254, 193.61650705337524, 197.66081285476685, 201.49179697036743, 205.19357800483704, 208.97110986709595, 213.19628596305847, 217.24929404258728, 221.06146907806396, 224.95390105247498, 229.3241250514984, 233.2227840423584, 237.20197200775146, 241.13403987884521, 245.3661630153656, 249.4290750026703, 253.66465401649475, 258.0823829174042, 262.7218029499054, 267.3874728679657, 271.3476278781891, 275.98510789871216, 280.05672693252563, 284.7069299221039, 288.99901008605957, 293.51549100875854, 297.38745498657227, 301.2563199996948, 305.29936385154724, 309.37054204940796, 313.48313903808594, 317.79451608657837, 322.55608797073364, 326.4949960708618, 330.71542596817017])
runtime.append([86.82124590873718, 91.04160499572754, 95.19278407096863, 99.32034397125244, 103.4765088558197, 107.67901706695557, 111.86226105690002, 115.39589405059814, 118.86516189575195, 122.31539797782898, 125.87774705886841, 129.49045991897583, 133.08521485328674, 136.68027091026306, 140.32491207122803, 143.98438787460327, 147.62968587875366, 151.50517201423645, 155.4433469772339, 159.15503191947937, 162.74135994911194, 166.35717296600342, 170.14332604408264, 174.10028195381165, 177.82677698135376, 181.68234086036682, 185.48014092445374, 189.37860298156738, 193.5907289981842, 197.264643907547, 201.0022509098053, 204.89533495903015, 208.61183905601501, 212.4743070602417, 216.61052799224854, 220.9965100288391, 225.12007403373718, 229.25894284248352, 233.0801899433136, 236.92572784423828, 240.8382499217987, 244.76363897323608, 249.21906304359436, 253.45885491371155, 257.8503198623657, 261.8198549747467, 266.18424892425537, 270.73065996170044, 275.31175684928894, 279.4530620574951, 283.44158697128296, 287.32920598983765, 292.0491008758545, 296.883229970932, 301.06921195983887, 305.1704559326172, 310.12547993659973, 314.25822591781616, 319.1861848831177, 323.0776219367981, 327.38884592056274])
runtime.append([87.53721904754639, 91.71324396133423, 95.86490106582642, 99.990238904953, 104.07824802398682, 108.25971293449402, 112.41233897209167, 115.87434792518616, 119.30591487884521, 122.80477809906006, 126.32498097419739, 129.85516500473022, 133.40196204185486, 136.95840191841125, 140.56162190437317, 144.1536819934845, 147.77122688293457, 151.51785492897034, 155.31338787078857, 158.97215700149536, 162.5326509475708, 166.0953369140625, 169.6819338798523, 173.36509704589844, 177.06679105758667, 180.99821209907532, 184.880441904068, 188.54394388198853, 192.30470609664917, 196.18251395225525, 199.8282129764557, 203.95090889930725, 207.60744094848633, 211.6174349784851, 215.67494702339172, 219.4221088886261, 223.54575204849243, 227.47228002548218, 231.7100670337677, 235.69919300079346, 239.41888308525085, 243.8151979446411, 247.71377801895142, 251.422945022583, 255.64129400253296, 259.44502091407776, 263.2103900909424, 267.6283938884735, 272.13961601257324, 276.11584186553955, 280.0808789730072, 283.9271900653839, 287.9791839122772, 292.13406705856323, 296.4141569137573, 300.6501519680023, 304.6629469394684, 309.2134790420532, 313.85185289382935, 318.39587903022766, 322.4422459602356])
runtime.append([87.53760290145874, 91.66621685028076, 95.80941486358643, 100.03377294540405, 104.22129893302917, 108.42688488960266, 112.65208292007446, 116.1096830368042, 119.53810095787048, 122.97238802909851, 126.54384088516235, 130.10245895385742, 133.675843000412, 137.3079810142517, 141.10489296913147, 144.71970582008362, 148.41302180290222, 152.09238696098328, 155.82326483726501, 159.48935890197754, 163.10072493553162, 166.77087497711182, 170.56009197235107, 174.3020749092102, 178.07772302627563, 181.83927702903748, 185.7325758934021, 189.7439568042755, 193.41492295265198, 197.40888690948486, 201.05963397026062, 204.9218418598175, 209.02252101898193, 212.74369096755981, 216.64769196510315, 220.89098596572876, 224.93334484100342, 229.1382339000702, 233.14367389678955, 237.34331488609314, 241.62991285324097, 246.15653800964355, 250.91113686561584, 254.77471089363098, 258.514652967453, 262.79553294181824, 266.69946098327637, 270.54966592788696, 274.4209189414978, 278.5250668525696, 282.4696378707886, 286.80742597579956, 291.3989019393921, 295.2334318161011, 299.71058893203735, 303.8408579826355, 308.4339678287506, 312.50388503074646, 316.5241370201111, 321.2596299648285, 325.4955749511719])

average_runtime = list(np.mean(np.matrix(runtime), axis=0).A1)


id = np.where(np.array(labels_all) == 88)[0][0]

print average_runtime[id]