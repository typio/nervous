export const print = async (t, decimals) => {
    console.log(JSON.stringify(await t.values(decimals)).replace(/],/g, '],\n '))
}
