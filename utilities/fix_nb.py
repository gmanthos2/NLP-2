import json

path = '/home/george/MScAI/NLP-2/Part_A_Embeddings.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        if not source:
            continue
        
        # fix top_ten_w2v_glove to return the dictionaries
        if source[0].startswith("def top_ten_w2v_glove(query_words):"):
            if not any("return w2v_results, glove_results" in line for line in source):
                source.append("\n")
                source.append("    return w2v_results, glove_results\n")
                
        # fix the call for query_words
        if len(source) >= 2 and source[0].startswith("query_words ="):
            for i, line in enumerate(source):
                if line.strip() == "top_ten_w2v_glove(query_words)":
                    source[i] = line.replace("top_ten_w2v_glove(query_words)", "w2v_results, glove_results = top_ten_w2v_glove(query_words)")
                    
        # fix the call for custom_words
        if len(source) >= 2 and source[0].startswith("custom_words ="):
            for i, line in enumerate(source):
                if "top_ten_w2v_glove" in line:
                    source[i] = "w2v_results_custom, glove_results_custom = top_ten_w2v_glove(custom_words)\n"
                    
        # fix the set logic for custom_words
        if len(source) >= 2 and source[0].startswith("for word in custom_words:"):
            for i, line in enumerate(source):
                if "w2v_results.get" in line:
                    source[i] = line.replace("w2v_results.get", "w2v_results_custom.get")
                if "glove_results.get" in line:
                    source[i] = line.replace("glove_results.get", "glove_results_custom.get")

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Fixed notebook!")
