import os

lines = list()

with open('./flickr8k/annotations/annotations_image_id_test.csv', 'r') as fr:
    
    lines = fr.readlines()
    print(len(lines))
    
    for line in lines[1:]:
        
        line_s = line.split(';')
        
        path = './flickr8k/images/test/' + line_s[0]
        
        if not os.path.exists(path):
            lines.remove(line)
    
    print(len(lines))
    
    
    
with open('./flickr8k/annotations/annotations_image_id_test.csv', 'w') as fw:
    
    for line in lines:
        
        fw.write(line)
        
        
        
print(len(lines))
    
