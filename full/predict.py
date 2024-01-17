from FEN import FEN
from chessboard_detection import *
import torch
import torchvision.transforms as transforms
from PIL import Image
from models import *


def occupation_classify(model, x):
    class_names = ['empty', 'occupied']
    with torch.no_grad():
        y = model(x)
        #display(torch.max(output, 1))
        prediction = torch.max(y, 1)[1]
    return prediction, [class_names[i] for i in prediction]



def predict(squares : np.ndarray, model_name = "CNN_80x80_2Conv_2Pool_2FC", model_saves_path = './scratch-cnn/modelsaves/'):
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
    
    model = torch.load(f"{model_saves_path}{model_name}.pth")
    transform = transforms.Compose([
        transforms.Resize((80, 80)),
        transforms.ToTensor()
        ])
    
    occupation_cnn_input = torch.empty([64, 3, 80, 80])
    for i, img in squares[:,[0,2]]:
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        occupation_cnn_input[i-1] = transform(img).reshape([1, 3, 80, 80])
    occupation_prediction = occupation_classify(model, occupation_cnn_input)[0]
    
    occupied_squares = squares[occupation_prediction==1]
    
    #np.column_stack([squares[:, ], occupation_prediction])
    
    return {coord: 'w_King' for coord, bbox in occupied_squares[:, [1,3]]}


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
    view = 'black'
    #optional:
    true_fen = "rn1qk2r/ppp1bppp/8/3pP2b/4n3/3B1N1P/PPP2PP1/RNBQR1K1"
    
    
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
    
    #predicted_fen = "8/8/8/4n1K1/1qk1P3/8/5N2/8 b - - 0 1"
    print_result(predicted_fen, true_fen)
        

if __name__ == "__main__":
    main()