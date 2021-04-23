device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
input_image = input_iamge.to(device)
input_label = input_label.to(device)