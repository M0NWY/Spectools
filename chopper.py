
# Use : chopper sample-length filename.wav

import sys
from pydub import AudioSegment

mspercut = int(sys.argv[1])
filename = sys.argv[2]




a = AudioSegment.from_wav(filename)

print( "info :")

print(a.__len__())

#print( a.frame_count())
length = int(a.__len__())
print ("choppin'")

for x in range(0,length,mspercut):

	chunk = a[x: (x + mspercut)]

	exfnstr = "chopped"

	exfnstr += str(x)

	exfnstr += ".wav"

	chunk.export(exfnstr,format="wav")

print ( "Done ?" )

