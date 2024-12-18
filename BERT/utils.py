import csv
import torch
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, mean_squared_error
from config import BATCH_SIZE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, input_ids, attention_mask, numeric_features, targets, optimizer, batch_size=BATCH_SIZE):
    model.train()
    total_loss = 0
    for i in range(0, input_ids.size(0), batch_size):
        batch_input_ids = input_ids[i:i + batch_size]
        batch_attention_mask = attention_mask[i:i + batch_size]
        batch_numeric_features = numeric_features[i:i + batch_size]
        batch_targets = targets[i:i + batch_size].to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_input_ids, batch_attention_mask, batch_numeric_features)
        # loss = F.l1_loss(outputs.squeeze(), batch_targets)
        loss = F.mse_loss(outputs.squeeze(), batch_targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / (input_ids.size(0) // batch_size)

def evaluate(model, input_ids, attention_mask, numeric_features, targets, batch_size=BATCH_SIZE):
    
    experimental_result = []
    
    try:
        model.eval()
        predictions, actuals = [], []
        with torch.no_grad():
            for i in range(0, input_ids.size(0), batch_size):
                batch_input_ids = input_ids[i:i + batch_size]
                batch_attention_mask = attention_mask[i:i + batch_size]
                batch_numeric_features = numeric_features[i:i + batch_size]
                batch_targets = targets[i:i + batch_size].to(device)
                
                outputs = model(batch_input_ids, batch_attention_mask, batch_numeric_features)
                predictions.extend(outputs.squeeze().cpu().numpy())
                actuals.extend(batch_targets.cpu().numpy())
        
        mae = mean_absolute_error(actuals, predictions)
        mse = mean_squared_error(actuals, predictions)
        
        # 保存實驗結果
        experimental_result.append({
            "mae": mae,
            "mse": mse,
        })
        
    except Exception as e:
        error_message = str(e)
        print(f"{error_message}")
        
    '''# 將結果寫入 CSV 檔案
    csv_filename = "BERT_results.csv"
    with open(csv_filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["mae", "mse"])
        writer.writeheader()
        writer.writerows(experimental_result)

    print(f"CSV file '{csv_filename}' has been created!")'''
    
    return mae, mse

def adjust_learning_rate(optimizer, current_loss, previous_loss, factor=0.9, min_lr=1e-6):
    if current_loss > previous_loss:
        for param_group in optimizer.param_groups:
            new_lr = max(param_group['lr'] * factor, min_lr)
            param_group['lr'] = new_lr
            print(f"Learning rate decreased to {new_lr:.6e}")