from FEN import FEN
from chessboard_detection import *
import torch
import torchvision.transforms as transforms
from PIL import Image
from models import *


def occupation_classify(model, x):
    model.eval()
    class_names = ['empty', 'occupied']
    with torch.no_grad():
        y = model(x)
        #display(torch.max(output, 1))
        prediction = torch.max(y, 1)[1]
    return prediction, [class_names[i] for i in prediction]

def pieces_classify(model, x):
    model.eval()
    class_names = ['b_Knight', 'w_Knight','w_Queen','b_Queen','b_Rook','w_Bishop','w_Rook','w_Pawn','b_King','w_King','b_Pawn','b_Bishop']
    with torch.no_grad():
        y = model(x)
        #display(torch.max(output, 1))
        prediction = torch.max(y, 1)[1]
    return prediction, [class_names[i] for i in prediction]


def predict(squares : np.ndarray, model_name = "CNN_80x80_2Conv_2Pool_2FC_manual_cpu_stopped", model2_name ='2-ResNet50', model_saves_path = './scratch-cnn/modelsaves2/'):
    """
    Parameters
    ----------
    squares : numpy.ndarray
        shape (64, 4), each square [square num, coord, square_img, bbox_img]
    model_name : str
        ["CNN_80x80_2Conv_2Pool_2FC"]
    model_saves_path : str
        path to saving directory of trained models
        
    Return
    ------
    dictiornary of occupied squares as pair <coords: piece>
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(f"{model_saves_path}{model_name}.pth")
    model2 = torch.load(f"{model_saves_path}{model2_name}.pth", map_location=device)


    occupation_cnn_input = torch.empty([64, 3, 80, 80])
    for i, img in squares[:,[0,2]]:
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        occupation_cnn_input[i-1] = model.transform(img).reshape([1, 3, 80, 80])
    occupation_prediction = occupation_classify(model, occupation_cnn_input)[0]
    
    occupied_squares = squares[occupation_prediction==1]
    
    pieces_cnn_input = torch.empty([len(occupied_squares), 3, model2.shape_input[0], model2.shape_input[1]])
    for i, img in enumerate(occupied_squares[:,3]):
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        pieces_cnn_input[i] = model2.transform(img).reshape([1, 3, 80, 160])

    pieces_prediction = pieces_classify(model2, pieces_cnn_input)[1]
    return {coord: piece for coord, piece in np.column_stack((occupied_squares[:,1], pieces_prediction))}
    


def result(predicted_fen, true_fen):
    return FEN.cmpFEN(predicted_fen, true_fen)


def print_result(predicted_fen, true_fen=None):
    print(f"{' RESULT ':#^32}")
    print(f"{'Predicted FEN:':<15}{predicted_fen}")
    FEN.print_from_FEN(predicted_fen)
    if true_fen is not None:
        print(f"{'True FEN:':<15}{true_fen}")
        FEN.print_from_FEN(true_fen)
    
    if true_fen is not None:    
        print()
        diff_dict = result(predicted_fen, true_fen)
        
        if len(diff_dict) == 0:
            print("No Errors\nAcc: 100%")
        else:
            print(f"Accuracy: {1-len(diff_dict)/64:.1%}")
            print(f"{len(diff_dict)} Errors:")
            print(f"{'Square':^8}|{'Predicted':^12}|{'Truth':^10}")
            for square, piece_pair in diff_dict.items():
                pred, true = piece_pair
                print(f"{square:^8}|{pred:^12}|{true:^10}")
    
    print(f"\n{'#'*32}\n")
    
    
    
def main():
    img_path = './input/0000.png'
    view = 'white'
    #optional:
    true_fen = "6k1/1b2q1P1/3Rp1p1/2n1B3/5P2/1P6/r1P1Q3/2K5"
    
    
    warpedBoardImg = board_detection(img_path, '0000')
    if warpedBoardImg is None:
        print('Error 1st preprocessing pass (Chessboard warping)')
        exit(-1)
    
    grid_squares = grid_detection(warpedBoardImg, view)
    if grid_squares is None:
        print('Error in 2nd preprocessing pass (Squares detection)')
        exit(-2)
    
    predicted_pos = predict(grid_squares)
    predicted_fen = FEN.dict_to_fen(predicted_pos)
    
    print_result(predicted_fen, true_fen)

        

if __name__ == "__main__":
    main()