import DiagObjects
import GaussDiag

g = DiagObjects.GaussDiag('O1-U2-O3+U1-O2-U3+')
g2 = DiagObjects.GaussDiag('O1-O2-O3+U2-U1-O4-U5+U3+U6+O5+U4-U7+U8+O6+O8+O7+')
s = DiagObjects.StringLink(['O1-U2-O3+U1-O2-U3+'])
s2 = DiagObjects.StringLink(['O4+O2+U4+U1+U3+', 'O1+U2+O3-'])
s3 = DiagObjects.StringLink(['O2+O3+O4-O1-', 'U4-U2+', 'U1-U3+'])
s4 = DiagObjects.StringLink(['O1+U1+O2-U3-O4+U5-', 'O3-O5+U6-O7-O8+U7-O6-U8+', 'U4+U9+OA+UB-UA+O9+U2-OB-'])
s5 = DiagObjects.StringLink(['O1+U2-O3-', 'O4+O2-U3-U4+U1+'])

DiagObjects.draw(s2, startingLabel= ['a','b'], arrowpoints = True)
print(GaussDiag.AIP(g))