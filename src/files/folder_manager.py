import glob, os, shutil

class FolderManager:
    def __init__(self, path):
        self.folder = path
        self.files = self._scan_files()
        self.work_file = self.files[0] if self.files else None

    def _scan_files(self):
        FILE_EXT = ['png', 'jpg', 'gif']
        files = []
        for ext in FILE_EXT:
            files.extend(glob.glob(os.path.join(self.folder, f'*.{ext}')))
        return files

    def get_work_file(self):
        return self.work_file

    def set_work_file(self, file_path):
        if file_path in self.files:
            self.work_file = file_path

    def get_files(self):
        return self.files

    def init_output_folder(self):
        output_folder = os.path.join(self.folder, '__OUTPUT_FILES__')
        if not os.path.isdir(output_folder):
            os.makedirs(output_folder)
        for src_file in self.files:
            out_file = os.path.join(output_folder, os.path.basename(src_file))
            if not os.path.isfile(out_file):
                shutil.copy(src_file, out_file)