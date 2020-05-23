import time


class Logger:
    def __init__(self, log_folder, a, b, c, steps, content, style):
        now = time.localtime()
        #self.log_path = log_folder + "a{:.1f}_b{:.2f}_c{}_{}_steps.txt".format(a, b, c, steps)
        self.log_path = log_folder + style.split(".")[0] + "_" + content.split(".")[0] + "_" + time.strftime("%Y-%m-%d_%H-%M-%S.txt", now)
        self.logfile = open(self.log_path, "w")
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", now)
        self.logfile.write(timestamp + "\n")
        self.logfile.write("Weights (content, style, variation) = ({:.1f}, {:.2f}, {})\n".format(a, b, c))
        self.logfile.write("Update steps: {}\n".format(steps))
        self.logfile.write("Content image: " + content + "\n")
        self.logfile.write("Style   image: " + style + "\n\n")

    def param_change(self, step, a, b, c, lr):
        self.logfile.write("\n" + "-" * 40 + "\n")
        self.logfile.write("Parameters changed at step {}: ".format(step) +
                           "weights (content, style, variation) = ({:.1f}, {:.2f}, {}) ; ".format(a, b, c) +
                           "learning rate = " + lr + "\n")

    def print_loss(self, step, loss, s_loss, c_loss):
        self.logfile.write("Step {:=5d} - total loss: {:=16.1f}\t[ S: {:=14.1f} ; C: {:=12.1f} ]\n".format(step, loss, s_loss, c_loss))

    def close(self):
        ts = time.strftime("%H:%M:%S", time.localtime(1))
        self.logfile.write("\nFinished at " + ts + "\n")
        self.logfile.close()
