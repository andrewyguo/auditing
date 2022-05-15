import time

def save_results(results, output_path, start_time):
    end_time = time.time()

    output = open(output_path, "a")

    
    output.write("Total time taken for this run: {} seconds\n".format(end_time - start_time))
    output.write("\n----\n")
    output.close()