# -*- coding: utf-8 -*-

"""
  2018-05-24
    jednostavan vizualizator trajektorija

    ulaz:
      datoteka s videom
      datoteka s trajektorijama

"""



import cv2
import time
import tkinter as tk   # za određivanje veličine ekrana kako bih smanjio prikaz



# frame od kojega kreće prikaz, tek ako je > 100, premotava
pocetakPrikaza = 10
debljinaOkvira = 1
step = 1

# ----------------
imeVidea = r"t7.mp4"
imeTrajektorija = r"trajektorije.txt"



class Trajektorije(object):

  def obradiTrajektoriju(self, trajektorija, cumulativeW, cumulativeH):


      # par minimalnih filtara na duljinu trajektorije i veličine BBoxa
      #if len(trajektorija) < 20:
      #    return

      averageW = cumulativeW/len(trajektorija)
      averageH = cumulativeH/len(trajektorija)

      aspectRatio = averageH/averageW

      if aspectRatio < 1 or aspectRatio > 4:
          return

      area = averageW*averageH

      if area < 100 or area > 1000:
          return

      for igrac in trajektorija:
          (frameID, playerID, teamID, x1, y1, w, h) = igrac

      if frameID >= len(self.frames):
          nadopuna = frameID + 1 - len(self.frames)
          self.frames.extend([[]]*nadopuna)
      self.frames[frameID].append((frameID, playerID, teamID, x1, y1, w, h))



  def __init__(self, imeDatoteke, faktorSkaliranja):
    self.frames = []

    with open(imeDatoteke) as dat:
      stanje = 0
      privremenaTrajektorija = []
      cumulativeW = 0
      cumulativeH = 0

      for redak in dat:

        rez = redak.rstrip().split()
        #print(rez)
        if len(rez) < 7:
          print("Nešto ne štima u retku: {}".format(redak))
          continue

        privremenaTrajektorija = []
        cumulativeW = 0
        cumulativeH = 0


        frameID = int(rez[0])
        playerID = int(rez[1])
        teamID = int(rez[2])

        minRow = int(int(rez[3])/faktorSkaliranja)
        maxRow = int(int(rez[4])/faktorSkaliranja)
        minCol = int(int(rez[5])/faktorSkaliranja)
        maxCol = int(int(rez[6])/faktorSkaliranja)

        # ostali podaci nas za ovu potrebu ne zanimaju


        y1 = maxRow
        x1 = minCol
        w = maxCol - minCol + 1
        h = maxRow - minRow + 1

        cumulativeH += h
        cumulativeW += w

        privremenaTrajektorija.append((frameID, playerID, teamID, x1, y1, w, h))
        self.obradiTrajektoriju(privremenaTrajektorija, cumulativeW, cumulativeH)

# =========================================================================================================

def iscrtajIgrace(indeks, frame):

    fontFace = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.4

    hf, wf, channels = frame.shape

    if indeks >= len(noveTrajektorije):
        return

    okvir = noveTrajektorije[indeks]
    #print(okvir)

    for igrac in okvir:
      (frameID, playerID, teamID, x1, y1, w, h) = igrac


      x2 = x1 + w  # desna
      y2 = y1 - h  # gornja (manji indeks)

      farbe = [(255, 255, 255),   # tim 0 -- suci
               (255, 255, 0),     # tim 1 -- golman plavi
               (255, 255, 0),     # tim 1 -- plavi
               (200, 0, 220),     # tim2 -- golman crveni
               (200, 0, 220)]     # tim2  -- crveni

      # uveo sam teamID
      boja = farbe[teamID]


      cv2.rectangle(frame, (x1, y1), (x2, y2), boja, debljinaOkvira)

      tekst = "{}".format(playerID)
      cv2.putText(frame, tekst, (x2, y2), fontFace, fontScale, boja)


    vrijeme = indeks/capFPS # sekundi
    minute = int(vrijeme/60)
    sekunde = vrijeme - 60*minute # float
    tekst = "frame: {:06} time: {:02}:{:05.2f}".format(indeks, minute, sekunde)

    cv2.rectangle(frame, (wf - 300, 5), (wf -5, 25), (0,0,0),-1) # -1 je filled
    cv2.putText(frame, tekst, (wf - 290, 20), fontFace, fontScale, (255, 255, 255))

def main():

  global faktorSkaliranja
  global capFPS

  root = tk.Tk()
  screen_width = root.winfo_screenwidth()
  screen_height = root.winfo_screenheight()
  print("{} x {}".format(screen_width, screen_height))

  cap = cv2.VideoCapture(imeVidea)
  inputW = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
  inputH = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
  capFPS = int(cap.get(cv2.CAP_PROP_FPS))

  "Dodano"
  fourcc = cv2.VideoWriter_fourcc(*'XVID')
  out_vid = cv2.VideoWriter('output.avi', fourcc, 25.0, (1600, 279))

  faktorSkaliranja = inputW/screen_width

  global noveTrajektorije
  start = time.time()

  trajektorije = Trajektorije(imeTrajektorija, faktorSkaliranja)
  noveTrajektorije = trajektorije.frames


  end = time.time()

  print("\nUčitavanje trajektorija: {} sekundi. \n".format(end - start))

  tekuciFrame = 0
  Pauza = True

  if pocetakPrikaza > 100:
    cap.set(cv2.CAP_PROP_POS_FRAMES, pocetakPrikaza)
    tekuciFrame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

  i = 0
  while(1):
    ret, frame = cap.read()
    if not ret:
        break

    tekuciFrame += 1

    if tekuciFrame%step != 0:
      continue


    if faktorSkaliranja != 1:
      #print(str(int(inputW/faktorSkaliranja)) + " " + str(int(inputH/faktorSkaliranja)))
      mala = cv2.resize(frame, (int(inputW/faktorSkaliranja), int(inputH/faktorSkaliranja)), interpolation = cv2.INTER_LINEAR)
      iscrtajIgrace(tekuciFrame, mala)
      if i<1000:
          out_vid.write(mala)
      if i==1000:
          out_vid.release()
      cv2.imshow("frame", mala)
    else:
      iscrtajIgrace(tekuciFrame, frame)
      cv2.imshow("frame", frame)



    if Pauza:
       k = cv2.waitKey(0) & 0xff
       if k == 27:
          break  #prekida program ako pritisnemo ESC (ASCII 27)
       elif k == ord("p") or k == ord("P"):
          Pauza = False

    else:
       k = cv2.waitKey(1) & 0xff
       if k == 27:
          break  #prekida program ako pritisnemo ESC (ASCII 27)
       elif k == 112:
          Pauza = True

    i+=1
  cv2.destroyAllWindows()
  cap.release()


if __name__ == "__main__":
  main()
