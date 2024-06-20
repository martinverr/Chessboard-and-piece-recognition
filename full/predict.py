from FEN import FEN
from chessboard_detection import *
import torch
from torch import nn
import torchvision.transforms as transforms
from PIL import Image
from models import *
from retrieval import retrieval
import sys
import json

# class_names = ['b_Knight', 'w_Knight','w_Queen','b_Queen','b_Rook','w_Bishop','w_Rook','w_Pawn','b_King','w_King','b_Pawn','b_Bishop'] resnet 50
class_names = ['b_Pawn','b_Bishop','b_Knight','b_Rook','b_Queen','b_King', 'w_King','w_Queen','w_Rook','w_Knight','w_Bishop','w_Pawn']

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
    # class_names = ['b_Knight', 'w_Knight','w_Queen','b_Queen','b_Rook','w_Bishop','w_Rook','w_Pawn','b_King','w_King','b_Pawn','b_Bishop'] resnet 50
    class_names = ['b_Pawn','b_Bishop','b_Knight','b_Rook','b_Queen','b_King', 'w_King','w_Queen','w_Rook','w_Knight','w_Bishop','w_Pawn']
    with torch.no_grad():
        y = model(x)
        #display(torch.max(output, 1))
        prediction = torch.max(y, 1)[1]
    return prediction, [class_names[i] for i in prediction]


def predict(squares : np.ndarray, model_name = "CNN_80x80_2Conv_2Pool_2FC_manual_cpu_stopped", model2_name ='2-ResNet50', model_saves_path = './scratch-cnn/modelsaves2/', ensamble = False):
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
    dictionary of occupied squares as pair <coords: piece>
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(f"{model_saves_path}{model_name}.pth", map_location=device )
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

    if ensamble:
        pieces_retrieval_fv = retrieval(imgs=pieces_cnn_input, model=model2, ensamble = True) 
        soft = nn.Softmax(dim=-1)
        pieces_predict = []
        with torch.no_grad():
            pieces = model2(pieces_cnn_input)
        for p, r in zip(pieces, pieces_retrieval_fv):
            softy = soft(p)
            sum_pr = softy + torch.tensor(r)
            sum_pr = torch.nn.functional.normalize(sum_pr, p=2, dim=0)
            pieces_predict.append(sum_pr)
        pieces_predict = torch.stack(pieces_predict)
        prediction = torch.max(pieces_predict, 1)[1]
        pieces_prediction = [class_names[i] for i in prediction]
    else:
        pieces_prediction = pieces_classify(model2, pieces_cnn_input)[1]
    
    pieces_retrieval_prediction = retrieval(imgs=pieces_cnn_input, model=model2) 

    return {coord: piece for coord, piece in np.column_stack((occupied_squares[:,1], pieces_prediction))}, {coord: piece for coord, piece in np.column_stack((occupied_squares[:,1], pieces_retrieval_prediction))}
    


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
    

def print_differencies_cnn_retr(cnn_prediction, retrieval_prediction):
    print("\nDifferencies between cnn and retrieval predictions:")
    some_differences = False
    for coord, pred in retrieval_prediction.items():
        if cnn_prediction[coord] != pred:
            some_differences = True
            print(coord, ' cnn: ', cnn_prediction[coord], ' retrieval: ', pred)
    if some_differences == False:
        print('No differences found')    

def return_accuracy(predicted_fen, true_fen=None):
    #print(f"{' RESULT ':#^32}")
    #print(f"{'Predicted FEN:':<15}{predicted_fen}")
    #FEN.print_from_FEN(predicted_fen)
    
    if true_fen is not None:    
        diff_dict = result(predicted_fen, true_fen)
        
        if len(diff_dict) == 0:
            #print("No Errors\nAcc: 100%")
            accuracy = 1
        else:
            accuracy = 1-len(diff_dict)/64
            # print(f"Accuracy: {1-len(diff_dict)/64:.1%}")
            print(f"{len(diff_dict)} Errors:")
            print(f"{'Square':^8}|{'Predicted':^12}|{'Truth':^10}")
            for square, piece_pair in diff_dict.items():
                pred, true = piece_pair
                print(f"{square:^8}|{pred:^12}|{true:^10}")
    
    #print(f"\n{'#'*32}\n")
    return accuracy

    
def main():
    # check command syntax
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print("Syntax error: predict.py <img path> <'white'|'black'> [fen]")
        exit(-1)

    # parse command
    img_path = sys.argv[1]
    view = sys.argv[2]
    if len(sys.argv) == 4:
        true_fen = sys.argv[3]
    else:
        true_fen = None

    # retrieve warped chessboard
    warpedBoardImg = board_detection(img_path, old_version=True, verbose_show=True)
    if warpedBoardImg is None:
        print('Error 1st preprocessing pass (Chessboard warping)')
        exit(-2)
    # maskrcnn warping
    warpedBoardImg = board_detection(img_path, old_version=False, verbose_show=True)
    if warpedBoardImg is None:
        print('Error 1st preprocessing pass (Chessboard warping)')
        exit(-2)

    # Square detection and extraction
    grid_squares = grid_detection(warpedBoardImg, view, verbose_show=True)
    if grid_squares is None:
        print('Error in 2nd preprocessing pass (Squares detection)')
        exit(-3)
    
    # predict chessboard position with cnn and retrieval
    cnn_prediction, retrieval_prediction = predict(grid_squares)
    
    # show results cnn prediction
    predicted_fen = FEN.dict_to_fen(cnn_prediction)
    print_result(predicted_fen, true_fen)

    # show results retrival prediction
    for coord, pred in retrieval_prediction.items():
        print(coord, ' --> ', pred)

    # show different prediction
    print_differencies_cnn_retr(cnn_prediction, retrieval_prediction)

def main_ensamble():

    ensamble = True

    img_paths = glob.glob('./test/**.png')
    trues_fen = []
    views = []
    for path in img_paths:
        path_json = path.replace('.png', '.json')
        with open(path_json, 'r') as path_json:
            json_data = json.load(path_json)
        white_turn = json_data['white_turn'] #true is white, false is black
        fen = json_data['fen']
        views.append('white' if white_turn == True else 'black')
        trues_fen.append(fen)
    
    accuracies_old = []
    err_board_detection_old = 0
    err_grid_detection_old = 0

    accuracies_new = []
    err_board_detection_new = 0
    err_grid_detection_new = 0

    
    for index, img_path in enumerate(img_paths):
        print(f"Img: {img_path}")
        # retrieve warped chessboard
        warpedBoardImg_old_version = board_detection(img_path, old_version=True, verbose_show=False)
        if warpedBoardImg_old_version is None:
            print('Error 1st preprocessing pass (Chessboard warping)')
            err_board_detection_old += 1
            continue
        # maskrcnn warping
        warpedBoardImg_new_version = board_detection(img_path, old_version=False, verbose_show=False)
        if warpedBoardImg_new_version is None:
            print('Error 1st preprocessing pass (Chessboard warping)')
            err_board_detection_new += 1
            continue

        # Square detection and extraction
        grid_squares_old_version = grid_detection(warpedBoardImg_old_version, views[index], verbose_show=False)
        if grid_squares_old_version is None:
            print('Error in 2nd preprocessing pass (Squares detection)')
            err_grid_detection_old += 1
            continue

        grid_squares_new_version = grid_detection(warpedBoardImg_new_version, views[index], verbose_show=False)
        if grid_squares_new_version is None:
            print('Error in 2nd preprocessing pass (Squares detection)')
            err_grid_detection_new += 1
            continue
        
        
        # predict chessboard position with cnn and retrieval
        cnn_prediction_old, _ = predict(grid_squares_old_version, ensamble=ensamble, model_name = "Occupancy_CNN_80x80_2Conv_2Pool_2FC", model2_name ="ResNet18_80x160", model_saves_path = './scratch-cnn/modelsaves3/')
        cnn_prediction_new, _ = predict(grid_squares_new_version, ensamble =ensamble, model_name = "Occupancy_CNN_80x80_2Conv_2Pool_2FC", model2_name ="ResNet18_80x160", model_saves_path = './scratch-cnn/modelsaves3/')

        if cnn_prediction_new is None or cnn_prediction_old is None:
            continue

        # show results cnn prediction
        predicted_fen_old = FEN.dict_to_fen(cnn_prediction_old)
        accuracies_old.append(return_accuracy(predicted_fen_old, trues_fen[index]))

        # show results cnn prediction
        predicted_fen_new = FEN.dict_to_fen(cnn_prediction_new)
        accuracies_new.append(return_accuracy(predicted_fen_new, trues_fen[index]))

    mean_accuracy_old = np.array(accuracies_old).mean()
    perfect_fen_old = accuracies_old.count(1)
    correct_fen_accuracy_old = perfect_fen_old/len(img_paths)

    print(f"--------------------- ENSEMBLE {ensamble} ----------------")
    print(f'Test on {len(img_paths)}')
    print(f'Number of accuracies: {len(accuracies_old)}')
    print(f'Number of error in board detection: {err_board_detection_old}')
    print(f'Number of error in grid detection: {err_grid_detection_old}')
    print(f'Number of total image discarded for error in board detection and grid detection: {err_grid_detection_old + err_board_detection_old}')
    print(f'Mean accuracy (only for not discarded images): {mean_accuracy_old}')
    print(f'Correct FEN accuracy (only for not discarded images): {perfect_fen_old} / {len(accuracies_old)} = {correct_fen_accuracy_old}')

    mean_accuracy_new = np.array(accuracies_new).mean()
    perfect_fen_new = accuracies_new.count(1)
    correct_fen_accuracy_new = perfect_fen_new/len(img_paths)

    print(f'Test on: {len(img_paths)}')
    print(f'Number of accuracies: {len(accuracies_new)}')
    print(f'Number of error in board detection: {err_board_detection_new}')
    print(f'Number of error in grid detection: {err_grid_detection_new}')
    print(f'Number of total image discarded for error in board detection and grid detection: {err_grid_detection_new + err_board_detection_new}')
    print(f'Mean accuracy (only for not discarded images): {mean_accuracy_new}')
    print(f'Correct FEN accuracy (only for not discarded images): {perfect_fen_new} / {len(accuracies_new)} = {correct_fen_accuracy_new}')
    

    return 0
if __name__ == "__main__":
    main_ensamble()