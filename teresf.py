# from config.trainer_config import SAC_trainer

# print(SAC_trainer.ENV_SAC)
# print(getattr(SAC_trainer,"ENV_SAC"))

from config.Config_loader import get_config

print(get_config("SAC","SAC1","continuous"))

# import argparse
# import ast

# def main():
#     # Create the parser
#     parser = argparse.ArgumentParser(description="Parse a dictionary from the command line")

#     # Add an argument for the dictionary
#     parser.add_argument('--dict', type=str, required=True, help="Dictionary in string format")

#     # Parse the command-line arguments
#     args = parser.parse_args()

#     # Convert the string representation of the dictionary to an actual dictionary
#     try:
#         input_dict = ast.literal_eval(args.dict)
#         if not isinstance(input_dict, dict):
#             raise ValueError
#     except (ValueError, SyntaxError):
#         print("Error: The provided string is not a valid dictionary.")
#         return

#     # Use the dictionary
#     print("Parsed dictionary:", input_dict)

# if __name__ == "__main__":
#     main()
