import glob
import os
import shutil
import easyocr
import numpy

from tkinter import messagebox as mb

# FolderData > FileData > TextData
class FolderData:
    
    def __init__(self, path):
        self.__folder = path
        self.__files = []   # FileData 객체를 의미
        self.__work_file = None
        self.__init_work_folder()

    def __init_work_folder(self):
        FILE_EXT = ['png', 'jpg', 'gif']
        target_files = []
        [target_files.extend(glob.glob(self.__folder + '/' + '*.' + e)) for e in FILE_EXT]

        for tf in target_files:
            self.__files.append(FileData(tf))

        if len(self.__files) == 0:
            return

        self.__work_file = self.__files[0]

    def get_work_file(self):
        return self.__work_file
    
    def set_work_file(self, target_file):
        self.__work_file = target_file
        
    def get_files(self):
        return self.__files     # FileData 반환
    
    def get_file_by_index(self, index):
        return self.__files[index]      # FileData 반환
    
    def get_files_as_string(self):
        file_strings = []
        for f in self.__files:
            file_strings.append(f.get_file_name())
        return file_strings     # FileData의 name 반환
    
    def get_folder_path(self):
        return self.__folder

class FileData:
    def __init__(self, file):
        self.__name = file
        self.__texts = []   # TextData 객체를 의미
        self.__is_ocr_executed = False
        self.__faces = None

    def get_file_name(self):
        return self.__name

    def clear_faces(self):
        self.__faces = None

    def get_faces(self):
        return self.__faces

    def get_faces_as_string(self):
        faces = []
        if self.__faces == None or len(self.__faces) == 0:
            return None
        for f in self.__faces:
            faces.append(f.get_name())
        return faces

    def set_faces(self, positions):
        """
        positions:
            얼굴이 detect 된 영역에 대한 좌표값을 list 가 입력된다.
            예를 들어 얼굴이 2개 검출 된 경우 아래와 같은 좌표값이 입력된다.
            [[216 116 195 195],[101 188 197 197]]
        """
        if positions is not None and type(positions) is numpy.ndarray and len(positions) > 0:
            self.__faces = []
            for index, p in enumerate(positions):
                self.__faces.append(FaceData('얼굴-'+"{:04d}".format(index+1), p))
        
    def is_ocr_executed(self):
        return self.__is_ocr_executed        
        
    def set_texts(self, ocr_texts):
        """
        param ocr_texts: 
            ocr_texts 은 아래와 같은 형태의 data 구조로 들어온다.
            [
                ([[24, 48], [345, 48], [345, 109], [24, 109]], '下载手机天猫APP', 0.8471784057548907), 
                ([[24, 130], [368, 130], [368, 204], [24, 204]], '享388元礼包', 0.9837027854197318), 
                ([[190, 306], [290, 306], [290, 336], [190, 336]], '立即扫码', 0.9933473467826843), 
                ([[160, 348], [334, 348], [334, 372], [160, 372]], '下载手机天猫APP领福利', 0.858102917437849)
            ]
        """
        self.__texts = []
        for t in ocr_texts:
            self.__texts.append(TextData(t[1], t[0]))
        self.__is_ocr_executed = True

    def get_texts(self):
        return self.__texts     # TextData 객체 반환
    
    def get_text_by_index(self, index):
        return self.__texts[index]      # TextData 객체 반환
    
    def get_texts_as_string(self):
        return [t.get_text() for t in self.__texts]     # TextData의 text 반환
    
    def get_positions_as_string(self):
        return [t.get_position_info() for t in self.__texts]     # TextData의 position_info 반환
    
    def get_text_as_string_by_index(self, index):
        ocr_text_list = [t.get_text() for t in self.__texts]
        return ocr_text_list[index]     # # TextData의 text 반환
    
    def get_rectangle_position_by_texts_index(self, index):
        """
        __texts[index]의 position 좌표값을 가져오는 함수입니다.

        param index (int): 
            __texts[index]를 통해 TextData 반환할 때 사용.

        returns:
            start_pos, end_pos:
            사각형 그리기에 필요한 왼쪽 상단 좌표(start_pos)와 오른쪽 하단 좌표(end_pos).
        """
        print ('[DataManager] get_rectangle_position_by_texts_index() called...')
        rectangle_position = self.get_text_by_index(index).get_position_info()
        start_pos, end_pos = rectangle_position[0], rectangle_position[2]
        return start_pos, end_pos

class FaceData:
    def __init__(self, name, position):
        self.__name = name
        self.__position = position

    def get_name(self):
        return self.__name

    def get_position_info(self):
        return self.__position
    
    def __str__(self):
        return self.__name

class TextData:
    def __init__(self, text, position):
        self.__text = text
        self.__tr_text = None
        """
        OCR return example: 
            [
                ([[24, 48], [345, 48], [345, 109], [24, 109]], '下载手机天猫APP', 0.8471784057548907), 
                ([[24, 130], [368, 130], [368, 204], [24, 204]], '享388元礼包', 0.9837027854197318), 
                ([[190, 306], [290, 306], [290, 336], [190, 336]], '立即扫码', 0.9933473467826843), 
                ([[160, 348], [334, 348], [334, 372], [160, 372]], '下载手机天猫APP领福利', 0.858102917437849)
            ]
        position example: [[24, 48], [345, 48], [345, 109], [24, 109]]
        """
        self.__position_info = position  # Example: [[24, 48], [345, 48], [345, 109], [24, 109]]
        
    def set_tr_text_with_position(self, tr_text, position):
        self.__tr_text = tr_text
        self.__position_info = position
    
    def get_text(self):
        return self.__text

    def get_position_info(self):
        return self.__position_info

    def get_tr_text(self):
        return self.__tr_text
        
    
class DataManager:
    folder_data = None
    # no need to reset, reload
    easyocr_reader = None

    def init():
        curr_path = os.getcwd()
        default_image_path = curr_path + os.sep + "image"
        DataManager.reset_work_folder(target_folder=default_image_path)

        # this needs to run only once to load the model into memory
        DataManager.easyocr_reader = easyocr.Reader(['ch_sim','en'])
        
    @classmethod
    def get_work_file(cls):
        return DataManager.folder_data.get_work_file() 
    
    @classmethod
    def set_work_file(cls, target_file):
        DataManager.folder_data.set_work_file(target_file)
        
    @classmethod
    def reset_work_folder(cls, target_folder='./image'):
        print ('[DataManager.reset] reset, target=', target_folder)
        target_path = os.path.abspath(target_folder)
        cls.folder_data = FolderData(target_path)
        cls.__init_output_folder(target_path)

    @classmethod
    def __init_output_folder(cls,target_folder):
        print ('[DataManager] initOutputFiles() called...')
        print ('[DataManager] initOutputFiles() : target_folder = ', target_folder)

        output_folder = os.path.join(target_folder + os.sep + '__OUTPUT_FILES__')
        print ('[DataManager] initOutputFiles() : output_folder = ', output_folder)

        # create output folder if not exist
        if os.path.isdir(output_folder) == False:
            os.makedirs(output_folder)
            print ('[DataManager] initOutputFiles() : output_folder newly created!')
        
        if target_folder == None or len(target_folder) == 0:
            print ('[DataManager] initOutputFiles() : no source files!')
            return
        
        # copy files to output folder if source image file doesn't exist in output folder
        target_images = [file_data.get_file_name() for file_data in cls.folder_data.get_files()]
        for src_file in target_images:
            src_file_name = os.path.basename(src_file)
            out_file = os.path.join(target_folder, '__OUTPUT_FILES__', src_file_name)
            if not os.path.isfile(out_file):
                shutil.copy(src_file, out_file)
                
    
    @classmethod
    def get_output_file(cls):
        print ('[DataManager] get_output_file() called...')

        out_file_dir = cls.folder_data.get_folder_path()
        out_file_name = os.path.basename(cls.folder_data.get_work_file().get_file_name())
        out_file = os.path.join(out_file_dir, '__OUTPUT_FILES__', out_file_name)
        
        return out_file
    
    @classmethod
    def save_output_file(cls, out_image):
        print ('[DataManager] save_output_file() called...')
        src_file = cls.get_work_file().get_file_name()
        
        # out_image는 PIL의 Image 객체
        if out_image is None:
            print ('[DataManager] save_output_file() : image is None, it can not be saved!')

        # out_file은 path
        out_file = cls.get_output_file()
        print ('[DataManager] save_output_file() : src_file=', src_file)
        print ('[DataManager] save_output_file() : out_file=', out_file)

        if out_file is not None:
            if out_file.lower().endswith(("png")) == False:
                out_image.convert("RGB").save(out_file)
            else:
                out_image.save(out_file)
            print('[DataManager] save_output_file(): Image saved successfully!')
            return True
        return False

    @classmethod
    def get_prev_file(cls):
        img_file=DataManager.get_work_file()
        print('[DataManager] getPrevImageFile() called!!...')
        for i in range(len(cls.folder_data.get_files())):
            print('[DataManager] getPrevImageFile() i=', i, cls.folder_data.get_files()[i].get_file_name())
            if cls.folder_data.get_files()[i].get_file_name() == img_file.get_file_name():
                if i != 0:
                    print('[DataManager] getPrevImageFile() - image found : ', cls.folder_data.get_files()[i-1].get_file_name())
                    cls.set_work_file(cls.folder_data.get_files()[i-1])
                    return cls.folder_data.get_files()[i-1]
                else:
                    break
        print('[DataManager] getPrevImageFile() - image not found!!')
        return None

    @classmethod
    def get_next_file(cls):
        img_file=DataManager.get_work_file()
        print ('[DataManager] getNextImageFile() called!!...')
        for i in range(len(cls.folder_data.get_files())):
            print ('[DataManager] getNextImageFile() i=', i, ', curr_file=', img_file, ', compare=', cls.folder_data.get_files()[i].get_file_name())
            if cls.folder_data.get_files()[i].get_file_name() == img_file.get_file_name():
                if (i+1) < len(cls.folder_data.get_files()):
                    print ('[DataManager] getNextImageFile() - image found : ', cls.folder_data.get_files()[i+1].get_file_name())
                    return cls.folder_data.get_files()[i+1]
                else:
                    break
        print ('[DataManager] getNextImageFile() - image not found!!')
        return None

    @classmethod
    def get_image_index(cls):
        print ('[DataManager] get_image_index() called!!...')
        img_file = cls.folder_data.get_work_file()
        target_file = cls.folder_data.get_files()
        
        for i in range(len(target_file)):
            if target_file[i] == img_file:
                return i
        return -1

    @classmethod
    def get_texts_from_image(cls):
        """
        OCR을 통해 이미지 내에 존재하는 텍스트를 읽고, 리스트 형태로 반환합니다.
        FolderData의 files[i]를 통해 아래의 작업을 수행합니다.
        FileData의 texts, ocr_executed를 update합니다.
        TextData의 text, posiont을 update합니다.

        return: 
            list: OCR 결과로 추출된 텍스트 리스트를 리턴. 
        """
        print ('[DataManager] get_texts_from_image() called!!...')
        
        # 확장자 포함 path 값
        img_file = cls.folder_data.get_work_file().get_file_name()
        # 현재 작업 이미지 FileData 객체
        file_index = cls.get_image_index()
        work_file = cls.folder_data.get_file_by_index(file_index)
        ocr_executed_texts_list = []

        if work_file.is_ocr_executed():
            print('[DataManager] get_texts_from_image() called!!... 이미 읽은 데이터입니다.')
            ocr_executed_texts_list = work_file.get_texts_as_string()
    
            return ocr_executed_texts_list
        
        # easyocr을 통해 읽은 text를 저장
        ocr_texts = cls.easyocr_reader.readtext(img_file)
        DataManager.folder_data.get_work_file().set_texts(ocr_texts)

        ocr_executed_texts_list = work_file.get_texts_as_string()
        
        # can not read texts in image
        if len(ocr_executed_texts_list) == 0:
            mb.showwarning("경고", "text를 찾을 수 없습니다.")
            return None

        return ocr_executed_texts_list