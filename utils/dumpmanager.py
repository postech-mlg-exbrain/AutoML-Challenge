import tempfile
import shutil
import os

class DumpManager(object):
    def __init__(self, dir=None):
        self.trash = set([])
        self.keep = set([])
        self.dir = dir if dir else tempfile.mkdtemp()
        if len(os.listdir(self.dir)) != 0:
            raise ValueError("tmp_dir should not contain any file")

    def alloc_file(self):
        fd, filename = tempfile.mkstemp(dir=self.dir)
        os.close(fd)
        self.trash.add(filename)
        return filename

    def keep_file(self, filename):
        if filename not in self.trash:
            raise ValueError("There's no such file to keep: %s" % filename)
        self.trash.remove(filename)
        self.keep.add(filename)

    def empty_trash(self):
        n = len(self.trash)
        for t in self.trash:
            os.remove(t)
        self.trash.clear()
        return n

    def trash_except(self, files):
        files = set(files)
        self.trash |= (self.keep - files)
        self.keep &= files

    def clean_up(self):
        shutil.rmtree(self.dir, ignore_errors=True)

