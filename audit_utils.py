import time

def save_results(results, output_path, start_time):
    end_time = time.time()

    output = open(output_path, "a")

    output.write("Audit Run Started at: {}\n".format(start_time))
    output.write("Total time taken for this run: {} seconds\n".format(end_time - start_time))

    thresh, eps, acc = results 

    output.write("At threshold={}, epsilon={}.".format(thresh, eps))
    output.write("The best accuracy at distinguishing poisoning is {}.".format(acc))

    output.write("\n----\n")
    output.close()