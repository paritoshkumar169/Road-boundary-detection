model_path = os.path.join('public', 'models', f'{model_type}.pt')

model = YOLO(model_path)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
is_video = input_path.lower().endswith(video_extensions)

if is_video:
    cap = cv2.VideoCapture(input_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        results = model.predict(frame, conf=confidence, imgsz=1408, verbose=False)
        
        if results.masks is not None:
            mask = results.masks.data.cpu().numpy()
            mask = (mask * 255).astype(np.uint8)
            mask = cv2.resize(mask, (width, height))
            
            if display_mode == "draw" or display_mode == "highlight":
                overlay = frame.copy()
                overlay[mask > 128] =[255]
                
                alpha = 0.3 if display_mode == "highlight" else 0.1
                frame = cv2.addWeighted(overlay, alpha, frame, 0.7, 0)
            
            if display_mode != "none" and display_mode != "highlight":
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(frame, contours, -1, (0, 0, 255), 2)
                
        out.write(frame)
        
    cap.release()
    out.release()
else:
    try:
        pil_image = Image.open(input_path)
        if pil_image.mode == 'RGBA':
            pil_image = pil_image.convert('RGB')
        
        frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
    except Exception as e:
        frame = cv2.imread(input_path)
        if frame is None:
            raise ValueError(f"Could not read image: {input_path}. Error: {str(e)}")
    
    height, width, _ = frame.shape
    
    results = model.predict(frame, conf=confidence, imgsz=1408, verbose=False)
    
    if results.masks is not None and len(results.masks) > 0:
        mask = results.masks.data.cpu().numpy()
        mask = (mask * 255).astype(np.uint8)
        mask = cv2.resize(mask, (width, height))
        
        if display_mode == "draw" or display_mode == "highlight":
            overlay = frame.copy()
            overlay[mask > 128] =[255]
            
            alpha = 0.3 if display_mode == "highlight" else 0.1
            frame = cv2.addWeighted(overlay, alpha, frame, 0.7, 0)
        
        if display_mode != "none" and display_mode != "highlight":
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(frame, contours, -1, (0, 0, 255), 2)
    
    output_ext = os.path.splitext(output_path).lower()
    if output_ext not in ['.jpg', '.jpeg', '.png']:
        output_path = os.path.splitext(output_path) + '.jpg'
        
    cv2.imwrite(output_path, frame)

return output_path
