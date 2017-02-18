import cPickle
import sys
import caffe

def save_filters(network_def, network_model, save_path):
    #print 'arg1', network_def
    #print 'arg2', network_model
    #print 'arg3', save_path

    net = caffe.Net(network_def, network_model, caffe.TEST)

    params = []
    for k,v in net.params.items():
        print k, type(v), len(v)

        vlist = [vt.data for vt in v]
        params.append((k, vlist))

    dc = dict(params)
    with open("./model2.pkl", 'w') as fp:
        cPickle.dump(dc, fp, -1)

    return

def main(argv):

    #print argv[0]
    #print argv[0].lower()
    if len(argv) == 0:
        print 'To save filters:'
        print '  Saves filters to mat files.'
        print '  Usage: python caffe_ftr.py --save-filters network_def network_model save_path'
        exit()

    cmd_str = argv[0].lower()

    if cmp(cmd_str, '--save-filters')==0:
        print 'command: save-filters'
        if len(argv) != 4:
            print '  Saves filters to mat files.'
            print '  Usage: python caffe_ftr.py --save-filters network_def network_model save_path'
            print '    (args are similar.)'
            exit()
        save_filters(argv[1], argv[2], argv[3])
    else:
        print 'Unknown command: %s' % (cmd_str,)
    return


if __name__ == '__main__':
    main(sys.argv[1:])