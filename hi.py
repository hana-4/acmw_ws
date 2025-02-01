import cv2 as cv 
import mediapipe as mp

#part 1
mp_hands = mp.solutions.hands
#this module contains hand tracking solutions, detects and tracks hand landmarks and connections. uses ml models to identify 21 3d landmarks 
mp_draw = mp.solutions.drawing_utils 
#to draw the landmarks and the connections identified on the images.
mp_draw_styles =  mp.solutions.drawing_styles 
#to get the landmark connection drawing on the image in the defauolt styling (color, thinckness ets)


#part 2 

def hands_movement (hand_landmarks): 
    #made a function takes, landmarks detected as the arguments 
    lm = hand_landmarks.landmark
    #hand_landmarks is an object and youre extracting the landmark attribute of it. list of 21 attributes. 
    #we'll have x,y,z coordinates for everylandmark detected. Thats how we store landmark positions. 
    tips = [lm[i].y for i in [8,12,16,20]]
    #created a list containing of y coordinates of the fingertips
    #the indices correspond to the 4 finger tips

    base = [lm[i-2].y for i in [8,12,16,20]]
    #indices correspond to the same fingertips as before 
    #we're accessing i-2 so it retrieves y coords of the joints below the tips thus the bases


    #making gestures/gesture detection 

    if all(tips[i]>base[i] for i in range(4)):
        return "rock"
    elif all(tips[i]<base[i] for i in range(4)): 
        return "paper"
    elif all(tips[0]<base[0] and tips[1]<base[1] and tips[2]>base[2] and tips[3]>base[3] for i in range (4)):
        return "scissors" 
    
    return "unknown"


#part 3, creating game variables
vid_obj = cv.VideoCapture(0)
#object init to read vdo frames. 
clock = 0
#keep track of the frame count
player1_move = player2_move = None 
gametext = ""
success = True
#weather something has been detected or not



#part 4

#init mediapipe's hand model 
#mp is also ml trained. you can decide how complex you want your model to be. The more complex the model, better the accuracy more the computation
with mp_hands.Hands(model_complexity = 1, 
                    min_detection_confidence = 0.7, 
                    min_tracking_confidence = 0.7) as hands : 

#below two set minimum confidence thresholds for detecting and tracking hands 
    while True : 
        ret, frame = vid_obj.read()

        frame = cv.resize(frame, (1280,720))

        if not ret or frame is None : 
            break

        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        #mediapipe expects rgb and open cv reads in bgr
        results = hands.process(frame)
        #processing the frame to detect and track the hands 
        frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        #converting back to bgr format for opencv

        if results.multi_hand_landmarks : 
            #list containing of all detected hands in current frame. so 'if' checks if any landmarks detected if list empty move on
            for hand_landmark in results.multi_hand_landmarks : 
                #loop iterates over each set of detected hand landmarks
                

                #drawing the landmarks and the connections
                mp_draw.draw_landmarks (
                    frame, 
                    hand_landmark, 
                    mp_hands.HAND_CONNECTIONS, 
                    mp_draw_styles.get_default_hand_landmarks_style(), 
                    mp_draw_styles.get_default_hand_connections_style()
                )
        frame = cv.flip(frame, 1)


        #part5 gamelogic
        #we make the game count on a 100 frame cycle 

        #frame 0 to 19 display players "ready" 
        if 0<=clock<20 : 
            success = True
            gametext = "ready?"
            #frame 20 to 59 --> countdown
        elif clock<30:
            gametext= "3.."
        elif clock<50: 
            gametext = "2.."
        elif clock<60: 
            gametext = "1, go!..."
        elif clock ==60: 
            #detect hand landmarks and players moves. If two hands are detected success is set to true else false
            hls = results.multi_hand_landmarks
            #retrieves list of detected hand landmarks from results object 
            if hls and len(hls)==2: # to check if two hands detected 
                player1_move = hands_movement(hls[0])  
                player2_move = hands_movement(hls[1])
                #determines players hand movemenets based on first and second set of landmarks 
            else:
                success = False
        elif clock<100:
            #frame 60 to 99 deteermined the game outcomes 
            if success : 
                #if hand detected 
                gametext = f"player 1 played : {player1_move}, Player 2 played: {player2_move}."
                if player2_move == player1_move :
                    gametext = f"{gametext} game is tied"
                elif player1_move == "paper" and player2_move == "rock" : 
                    gametext=f"{gametext} Player 1 Wins"
                elif player1_move == "rock" and player2_move == "scissors" : 
                    gametext = f"{gametext} Player 1 Wins"
                elif player1_move == "scissors" and player2_move =="paper" : 
                    gametext=f"{gametext} Player 1 Wins"
                else:
                    gametext=f"{gametext}Player 2 wins"
            else: 
                gametext = "didnt play properly"

        cv.putText(frame,f"Clock:{clock}",(50,50),cv.FONT_HERSHEY_PLAIN,2,(0,255,255),2,cv.LINE_AA)
        cv.putText(frame,gametext,(50,80),cv.FONT_HERSHEY_PLAIN,2,(0,255,255),2,cv.LINE_AA)
        clock=(clock+1)%100
        cv.putText(frame, "Press 'Q' to Quit", (50, 120), cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2, cv.LINE_AA)

        cv.imshow('frame', frame)

        if cv.waitKey(10) & 0xFF == ord('q'):
            print("Exiting Program...")
            break
vid_obj.release()
cv.destroyAllWindows()
        




