import os, glob
from FEN import FEN
from chessboard_detection import *


def predict():
    return {}


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
    #predicted_pos = predict()
    #predicted_fen = FEN.dict_to_fen(predicted_pos)
    
    predicted_fen = "8/8/8/4n1K1/1qk1P3/8/5N2/8 b - - 0 1"
    print_result(predicted_fen, "8/8/8/4p1K1/2k1P3/8/5N2/8 b - - 0 1")
        

if __name__ == "__main__":
    main()