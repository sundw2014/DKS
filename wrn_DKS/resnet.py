import torch
import torch.nn.functional as F
import utils


def resnet(depth, width, num_classes, dropout):
    assert (depth - 4) % 6 == 0, 'depth should be 6n+4'
    n = (depth - 4) // 6
    widths = [int(v * width) for v in (16, 32, 64)]

    def gen_block_params(ni, no):
        return {
            'conv0': utils.conv_params(ni, no, 3),
            'conv1': utils.conv_params(no, no, 3),
            'bn0': utils.bnparams(ni),
            'bn1': utils.bnparams(no),
            'convdim': utils.conv_params(ni, no, 1) if ni != no else None,
        }

    def gen_group_params(ni, no, count):
        return {'block%d' % i: gen_block_params(ni if i == 0 else no, no)
                for i in range(count)}

    flat_params = utils.cast(utils.flatten({
        'conv0': utils.conv_params(3, 16, 3),
        'group0': gen_group_params(16, widths[0], n),
        'group1': gen_group_params(widths[0], widths[1], n),
        'group2': gen_group_params(widths[1], widths[2], n),
        'bn': utils.bnparams(widths[2]),
        'fc': utils.linear_params(widths[2], num_classes),
        # ep1
        'ep1_group1': gen_group_params(widths[0], widths[1], n),
        'ep1_group2': gen_group_params(widths[1], widths[2], n//2),
        'ep1_bn': utils.bnparams(widths[2]),
        'ep1_fc': utils.linear_params(widths[2], num_classes),
        # ep2
        'ep2_group2': gen_group_params(widths[1], widths[2]*2, n),
        'ep2_bn': utils.bnparams(widths[2]*2),
        'ep2_fc': utils.linear_params(widths[2]*2, num_classes),
    }))

    utils.set_requires_grad_except_bn_(flat_params)

    def block(x, params, base, mode, stride):
        o1 = F.relu(utils.batch_norm(x, params, base + '.bn0', mode), inplace=True)
        y = F.conv2d(o1, params[base + '.conv0'], stride=stride, padding=1)
        o2 = F.relu(utils.batch_norm(y, params, base + '.bn1', mode), inplace=True)
        if dropout:
            print('dropout')
            o2 = F.dropout(o2, 0.3, mode, inplace=False)
        z = F.conv2d(o2, params[base + '.conv1'], stride=1, padding=1)
        if base + '.convdim' in params:
            return z + F.conv2d(o1, params[base + '.convdim'], stride=stride)
        else:
            return z + x

    def group(o, params, base, mode, stride, _n=n):
        for i in range(_n):
            o = block(o, params, '%s.block%d' % (base,i), mode, stride if i == 0 else 1)
        return o

    def f(input, params, mode):
        x = F.conv2d(input, params['conv0'], padding=1)
        g0 = group(x, params, 'group0', mode, 1)
        g1 = group(g0, params, 'group1', mode, 2)
        g2 = group(g1, params, 'group2', mode, 2)
        o = F.relu(utils.batch_norm(g2, params, 'bn', mode))
        o = F.avg_pool2d(o, 8, 1, 0)
        o = o.view(o.size(0), -1)
        o = F.linear(o, params['fc.weight'], params['fc.bias'])

        ep1_g1 = group(g0, params, 'ep1_group1', mode, 2)
        ep1_g2 = group(ep1_g1, params, 'ep1_group2', mode, 2, _n=n//2)
        ep1_o = F.relu(utils.batch_norm(ep1_g2, params, 'ep1_bn', mode))
        ep1_o = F.avg_pool2d(ep1_o, 8, 1, 0)
        ep1_o = ep1_o.view(ep1_o.size(0), -1)
        ep1_o = F.linear(ep1_o, params['ep1_fc.weight'], params['ep1_fc.bias'])

        ep2_g2 = group(g1, params, 'ep2_group2', mode, 2)
        ep2_o = F.relu(utils.batch_norm(ep2_g2, params, 'ep2_bn', mode))
        ep2_o = F.avg_pool2d(ep2_o, 8, 1, 0)
        ep2_o = ep2_o.view(ep2_o.size(0), -1)
        ep2_o = F.linear(ep2_o, params['ep2_fc.weight'], params['ep2_fc.bias'])

        return (ep1_o, ep2_o, o)

    return f, flat_params
